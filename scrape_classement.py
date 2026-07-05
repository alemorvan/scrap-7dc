#!/usr/bin/env python3
"""
Scrape les classements depuis https://c7dc.ffaviron.fr/classement/epreuves
et génère un rapport PDF.

Usage:
    python scrape_classement.py
    python scrape_classement.py --categorie "H 40-49 TC" --no-extra --nom "ANTOINE LE MORVAN"
"""

import argparse
import json
import re
import sys
import textwrap
import time
from datetime import datetime

import numpy as np
import requests
from bs4 import BeautifulSoup

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

# ── Constantes ───────────────────────────────────────────────────────────────

BASE_URL   = "https://c7dc.ffaviron.fr/classement/epreuves"
FETCH_DELAY = 1
DOC_TITLE  = "Analyseur de performance C7DC et axes de progression"

# Filtre des outliers dans les graphiques.
# Cutoff = user_result ± (gap_entre_leader_et_user × CHART_OUTLIER_GAP_FACTOR).
# 1.0 = exclure tout ce qui est au-delà du même écart que leader→user.
# 0.5 = plus serré (exemple de l'utilisateur), 2.0 = plus permissif.
# None = pas de filtre.
CHART_OUTLIER_GAP_FACTOR = 0.5

EXTRA_LABELS = {
    "Printemps", "Été", "Automne", "Hiver",
    "21097 m", "42195 m",
    "24 h L.Team", "24 h S.Team", "24 h Solo", "24 h Tandem",
    "100 km L.Team", "100 km S.Team", "100 km Solo", "100 km Tandem",
}

# Palette bleue moderne
C = {
    "bg":           "#F4F8FE",
    "banner_bg":    "#1B3F7A",
    "banner_text":  "#FFFFFF",
    "banner_sub":   "#9EC5F0",
    "header_bg":    "#2E75B6",
    "header_text":  "#FFFFFF",
    "row_even":     "#EBF3FC",
    "row_odd":      "#FFFFFF",
    "user_bg":      "#D6EAFF",
    "user_text":    "#0D2D5E",
    "battus":       "#5BAD6F",
    "rang":         "#2E75B6",
    "curve":        "#2E75B6",
    "dot":          "#5BA5D5",
    "user_dot":     "#D62728",
    "grid":         "#D0E6F5",
    "text":         "#1C2833",
    "text_light":   "#7F8C8D",
    "border":       "#B0CCEB",
    "sep":          "#2E75B6",
}

# Difficulté relative de progresser de 1% par type d'épreuve.
# Court/explosif = difficile (×3), long/aérobie = plus accessible (×1.1).
EFFORT_MULTIPLIER = {
    "100 m":   3.0,
    "500 m":   2.5,
    "1000 m":  2.0,
    "2000 m":  1.8,
    "5000 m":  1.5,
    "10000 m": 1.2,
    "1 min":   2.5,
    "4 min":   1.8,
    "30 min":  1.3,
    "60 min":  1.1,
}

# Zones énergétiques C7DC (ordre du plus court au plus long).
# Clé → (set d'épreuves, label affiché)
ENERGY_ZONES = [
    ("Sprint",    {"100 m"},                        "Sprint pur"),
    ("Puissance", {"1 min", "500 m"},               "Puissance courte"),
    ("VO2max",    {"4 min", "1000 m"},              "VO2max"),
    ("Seuil",     {"2000 m", "5000 m", "30 min"},   "Seuil lactique"),
    ("Endurance", {"10000 m", "60 min"},            "Endurance fondamentale"),
]

# Ordre canonique d'affichage des épreuves (durée croissante).
EVENT_ORDER = [
    "100 m", "1 min", "500 m", "1000 m", "4 min",
    "2000 m", "5000 m", "30 min", "10000 m", "60 min",
]

# Marges (coordonnées normalisées A4 portrait, 1 cm ≈ 0.048)
MARGIN_L = 0.055   # bord gauche
MARGIN_R = 0.945   # bord droit
# Largeurs de retour à la ligne selon la taille de police (empirique DejaVu Sans A4)
WRAP = {10: 93, 9.5: 98, 9: 103, 8.5: 109}

# ── Scraping ─────────────────────────────────────────────────────────────────

def get_session():
    s = requests.Session()
    s.headers["User-Agent"] = (
        "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
    )
    return s


def fetch_page(session, epreuve_id, delay=False):
    if delay:
        time.sleep(FETCH_DELAY)
    resp = session.get(f"{BASE_URL}?epreuve_id={epreuve_id}", timeout=15)
    resp.raise_for_status()
    return resp.text


def get_epreuves(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    sel = soup.find("select", {"name": "epreuve_id"})
    if not sel:
        return {}
    return {
        o.get("value", "").strip(): o.get_text(strip=True)
        for o in sel.find_all("option")
        if o.get("value", "").strip()
    }


def find_all_classement_tables(html):
    """Retourne toutes les tables Classement/Nom/Résultat (une par groupe H/F)."""
    soup = BeautifulSoup(html, "html.parser")
    tables = []
    for table in soup.find_all("table", class_="table"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if headers == ["Classement", "Nom", "Résultat"]:
            tables.append(table)
    return tables


def parse_table(table, categorie_filter=None):
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 3:
            continue
        classement = cells[0].get_text(strip=True)
        nom_cell = cells[1]
        small = nom_cell.find("small", class_="text-muted")
        categorie = small.get_text(strip=True) if small else ""
        if small:
            small.extract()
        nom = nom_cell.get_text(strip=True)
        if categorie_filter and categorie.lower() != categorie_filter.lower():
            continue
        resultat = cells[2].get_text(strip=True)
        rows.append((classement, nom, categorie, resultat))
    return rows


def parse_all_tables(html, categorie_filter=None):
    """Parse toutes les tables H+F d'une page, avec filtre catégorie optionnel."""
    rows = []
    for table in find_all_classement_tables(html):
        rows.extend(parse_table(table, categorie_filter=categorie_filter))
    return rows


def auto_detect_category(session, epreuves, nom_lower):
    """
    Cherche nom_lower dans les données brutes (sans filtre catégorie) pour
    déduire automatiquement sa catégorie. Essaie les épreuves dans l'ordre
    jusqu'à trouver.
    """
    for eid, label in epreuves.items():
        try:
            html = fetch_page(session, eid)
            for row in parse_all_tables(html):
                if row[1].lower() == nom_lower:
                    return row[2]  # catégorie trouvée
        except Exception:
            continue
    return None


def is_competition(label):
    l = label.lower()
    return l.startswith("compétition") or l.startswith("competition")


def classify_event(label):
    """'distance' → résultat en mètres (1 min…) / 'time' → résultat en temps (100 m…)"""
    if re.search(r"\bmin\b", label, re.IGNORECASE):
        return "distance"
    if re.match(r"^\d+\s*m$", label):
        return "time"
    return None


def parse_result(value):
    if not value or value == "-":
        return None
    v = str(value).strip()
    if ":" in v:
        parts = v.split(":")
        try:
            if len(parts) == 2:
                return round(float(parts[0]) * 60 + float(parts[1]), 2)
            if len(parts) == 3:
                return round(float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2]), 2)
        except (ValueError, IndexError):
            return None
    try:
        return float(v)
    except ValueError:
        return None


def build_output_filename(categorie, no_extra, nom):
    parts = ["classements"]
    if categorie:
        parts.append(re.sub(r"[^\w\-]", "_", categorie).strip("_"))
    if nom:
        parts.append(re.sub(r"[^\w\-]", "_", nom).strip("_"))
    if no_extra:
        parts.append("no_extra")
    return "_".join(parts) + ".pdf"


# ── Analyse et recommandations ───────────────────────────────────────────────

def compute_analysis(event_data, nom_lower):
    """
    Pour chaque épreuve, calcule :
      - score % actuel, points actuels
      - densité : concurrents dans un écart de 5% juste devant l'utilisateur
      - amélioration % nécessaire pour gagner 10 places
      - score d'opportunité ajusté par la difficulté relative de l'épreuve
    Retourne la liste triée par opportunité décroissante.
    """
    results = []

    for label, rows in event_data.items():
        event_type = classify_event(label)

        user_rank, user_val = None, None
        for i, row in enumerate(rows, 1):
            if nom_lower and row[1].lower() == nom_lower:
                user_rank = i
                user_val  = parse_result(row[3])
                break

        if user_rank is None or user_val is None or user_val == 0:
            continue

        total          = len(rows)
        is_dist        = (event_type == "distance")
        current_pct    = (total - user_rank) / total * 100
        current_points = total - user_rank

        # Résultats valides avec leur rang (1-indexé)
        ranked = [
            (r + 1, parse_result(row[3]))
            for r, row in enumerate(rows)
            if parse_result(row[3]) is not None
        ]

        # Concurrents strictement au-dessus de l'utilisateur
        above_vals = [v for r, v in ranked if r < user_rank]

        # Densité à 5% : combien sont dans l'écart de 5% au-dessus
        if is_dist:
            density_5 = sum(1 for v in above_vals if v <= user_val * 1.05)
        else:
            density_5 = sum(1 for v in above_vals if v >= user_val * 0.95)

        # Amélioration % nécessaire pour gagner 10 places
        target_rank = max(1, user_rank - 10)
        target_val  = next((v for r, v in ranked if r == target_rank), None)
        if target_val:
            if is_dist:
                improvement_pct = max(0.0, (target_val - user_val) / user_val * 100)
            else:
                improvement_pct = max(0.0, (user_val - target_val) / user_val * 100)
        else:
            improvement_pct = 0.0

        # Score d'opportunité brut
        raw = (
            density_5 * 4.0 +                      # densité proche = gains immédiats
            max(0.0, 50 - current_pct) * 0.8 +     # marge de progression (bas % = plus de potentiel)
            max(0.0, 8 - improvement_pct) * 5.0    # petit effort requis = meilleure opportunité
        )
        # Ajusté par la difficulté relative : épreuves longues favorisées
        effort  = EFFORT_MULTIPLIER.get(label, 1.5)
        opp     = raw / effort

        results.append({
            "label":           label,
            "user_rank":       user_rank,
            "total":           total,
            "current_pct":     current_pct,
            "current_points":  current_points,
            "density_5pct":    density_5,
            "improvement_pct": improvement_pct,
            "opportunity":     opp,
        })

    return sorted(results, key=lambda x: x["opportunity"], reverse=True)


def _build_recs(analysis, total_points):
    """Génère les blocs de recommandations (titre, corps) basés sur l'analyse."""
    recs = []
    best  = max(analysis, key=lambda x: x["current_pct"])
    worst = min(analysis, key=lambda x: x["current_pct"])
    top   = analysis[0]

    # 1. Bilan global
    recs.append((
        "Bilan global",
        f"Tu totalises {total_points} points bonus sur {len(analysis)} épreuve(s) analysée(s). "
        f"Ton meilleur score relatif est sur {best['label']} "
        f"({best['current_pct']:.0f}%, rang {best['user_rank']}/{best['total']}). "
        f"L'épreuve où tu as le plus de marge est {worst['label']} "
        f"({worst['current_pct']:.0f}%, rang {worst['user_rank']}/{worst['total']})."
    ))

    # 2. Priorité 1 — gain rapide (densité élevée)
    a = top
    gains = min(10, a["density_5pct"])
    recs.append((
        f"Priorité n°1 — {a['label']} (gain rapide de points)",
        f"{a['density_5pct']} concurrent(s) se trouvent dans un écart de 5% "
        f"juste au-dessus de toi (rang {a['user_rank']}/{a['total']}, score {a['current_pct']:.0f}%). "
        f"Avec seulement {a['improvement_pct']:.1f}% d'amélioration, tu gagnerais ~10 places "
        f"et jusqu'à {gains} points bonus supplémentaires. "
        f"C'est l'épreuve avec le meilleur retour sur investissement à l'entraînement."
    ))

    # 3. Priorité 2
    if len(analysis) > 1:
        a2 = analysis[1]
        recs.append((
            f"Priorité n°2 — {a2['label']}",
            f"Score actuel {a2['current_pct']:.0f}% (rang {a2['user_rank']}/{a2['total']}). "
            f"{a2['density_5pct']} concurrent(s) proches au-dessus de toi. "
            f"Effort pour +10 places : {a2['improvement_pct']:.1f}%. "
            f"Bon potentiel avec un travail ciblé."
        ))

    # 4. Profil athlétique (endurance vs explosivité)
    long_ev  = [a for a in analysis if a["label"] in {"30 min", "60 min", "10000 m", "5000 m"}]
    short_ev = [a for a in analysis if a["label"] in {"100 m", "500 m", "1 min"}]
    if long_ev and short_ev:
        long_avg  = sum(x["current_pct"] for x in long_ev)  / len(long_ev)
        short_avg = sum(x["current_pct"] for x in short_ev) / len(short_ev)
        if long_avg - short_avg > 5:
            recs.append((
                "Profil : orientation endurance",
                f"Tes scores sur les épreuves longues ({long_avg:.0f}% en moyenne) dépassent "
                f"les courtes ({short_avg:.0f}%). Ton profil est aérobie. "
                f"Un travail en puissance maximale (intervalles courts, sprints) améliorerait "
                f"significativement tes résultats sur les épreuves explosives."
            ))
        elif short_avg - long_avg > 5:
            recs.append((
                "Profil : orientation explosivité",
                f"Tes scores sur les épreuves courtes ({short_avg:.0f}%) dépassent "
                f"les longues ({long_avg:.0f}%). Ton profil est puissance/explosif. "
                f"Des séances longues à faible intensité (endurance fondamentale > 45 min) "
                f"amélioreraient tes performances sur les épreuves aérobies."
            ))

    # 5. Note méthodologique
    recs.append((
        "Note sur la difficulté relative",
        "Progresser de 1% sur une épreuve courte (100 m, 500 m) est nettement plus difficile "
        "que sur une épreuve longue (30 min, 60 min) : les épreuves aérobies répondent mieux "
        "à l'entraînement en volume, tandis que les épreuves explosives nécessitent un travail "
        "technique et neuromusculaire très spécifique. Ce facteur est intégré dans le score "
        "de priorité du tableau ci-dessus."
    ))

    return recs


# ── Composants visuels ────────────────────────────────────────────────────────

def setup_style():
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          9,
        "figure.facecolor":   C["bg"],
        "axes.facecolor":     C["bg"],
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   True,
        "axes.spines.bottom": True,
        "axes.grid":          True,
        "grid.color":         C["grid"],
        "grid.linewidth":     0.5,
        "grid.alpha":         0.8,
        "xtick.color":        C["text"],
        "ytick.color":        C["text"],
        "text.color":         C["text"],
    })


def add_banner(fig, title, subtitle=""):
    """Bandeau bleu en haut de la page."""
    from matplotlib.patches import Rectangle
    # Rectangle au niveau figure — rendu fiable dans le PDF
    rect = Rectangle((0, 0.935), 1.0, 0.065,
                      transform=fig.transFigure,
                      facecolor=C["banner_bg"], edgecolor="none",
                      clip_on=False, zorder=1)
    fig.add_artist(rect)
    fig.text(0.5, 0.968, title,
             ha="center", va="center", fontsize=12, fontweight="bold",
             color=C["banner_text"], zorder=2)
    if subtitle:
        fig.text(0.5, 0.942, subtitle,
                 ha="center", va="center", fontsize=8,
                 color=C["banner_sub"], zorder=2)


def add_footer(fig, text):
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.03])
    ax.set_facecolor(C["bg"])
    ax.axis("off")
    ax.axhline(0.85, xmin=0.04, xmax=0.96, color=C["border"], linewidth=0.6)
    ax.text(0.5, 0.25, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=6.5, color=C["text_light"])


def _sort_events(event_data):
    """Trie event_data selon EVENT_ORDER, puis les épreuves inconnues par ordre alpha."""
    def key(label):
        try:
            return (0, EVENT_ORDER.index(label))
        except ValueError:
            return (1, label)
    return dict(sorted(event_data.items(), key=lambda kv: key(kv[0])))


def _format_gap_to_first(rows, user_rank, event_type):
    """Retourne une chaîne 'à +Xs / +Xm du 1er' ou '1er !' si rang 1."""
    if user_rank is None:
        return None
    if user_rank == 1:
        return "1er !"
    first_val = parse_result(rows[0][3])
    user_val  = parse_result(rows[user_rank - 1][3])
    if first_val is None or user_val is None:
        return None
    if event_type == "distance":
        gap = first_val - user_val
        if gap <= 0:
            return "1er !"
        return f"+{gap:.0f} m derrière le 1er"
    else:
        gap = user_val - first_val
        if gap <= 0:
            return "1er !"
        if gap >= 3600:
            h, rem = divmod(gap, 3600)
            m, s   = divmod(rem, 60)
            return f"+{int(h)}h{int(m):02d}m{s:02.0f}s derrière le 1er"
        elif gap >= 60:
            m, s = divmod(gap, 60)
            return f"+{int(m)}min {s:02.0f}s derrière le 1er"
        else:
            return f"+{gap:.1f}s derrière le 1er"


def _draw_info_card(fig, x0, card_y, w, card_h,
                    user_rank, total, pct, gap_str, improvement_pct, priority_rank):
    """Carte d info colonne unique : rang, %, ecart, effort + badge pill au-dessus."""
    from matplotlib.patches import FancyBboxPatch

    badge_colors = {1: "#D62728", 2: "#E87722", 3: "#2E75B6"}
    badge_labels = {1: "1re priorite", 2: "2e priorite", 3: "3e priorite"}
    has_badge = priority_rank and priority_rank in badge_colors

    badge_band = 0.030 if has_badge else 0.0
    real_card_h = card_h - badge_band
    real_card_y = card_y

    # Badge pill au-dessus
    if has_badge:
        fig.text(x0 + w / 2, real_card_y + real_card_h + badge_band / 2,
                 f"\u2605  {badge_labels[priority_rank]}",
                 ha="center", va="center",
                 fontsize=8, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.38",
                           facecolor=badge_colors[priority_rank],
                           edgecolor="none"))

    # Carte arrondie
    ax = fig.add_axes([x0, real_card_y, w, real_card_h])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch(
        (0.02, 0.03), 0.96, 0.94,
        boxstyle="round,pad=0.035",
        facecolor="#F0F6FF",
        edgecolor=C["banner_bg"],
        linewidth=1.6,
    ))

    # Colonne unique centree
    cx = 0.50
    y = 0.86

    ax.text(cx, y, f"#{user_rank} / {total}",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color=C["banner_bg"])
    y -= 0.24

    ax.text(cx, y, f"{pct:.0f} %",
            ha="center", va="top", fontsize=22, fontweight="bold",
            color=C["user_dot"])
    y -= 0.30

    if gap_str:
        ax.text(cx, y, gap_str,
                ha="center", va="top", fontsize=7.5, color=C["text"])
        y -= 0.22

    if improvement_pct is not None and improvement_pct > 0:
        ax.text(cx, y, f"Effort +10 places : {improvement_pct:.1f}%",
                ha="center", va="top", fontsize=7.5, color=C["text_light"])
    elif improvement_pct == 0:
        ax.text(cx, y, "\u2714 Deja dans le top 10",
                ha="center", va="top", fontsize=7.5, color=C["battus"])


def draw_table(ax, col_labels, rows, highlight_rows=None, col_widths=None, fontsize=7.5):
    """Table matplotlib stylisée sur un axes."""
    ax.axis("off")
    if not rows:
        ax.text(0.5, 0.5, "Aucune donnée", ha="center", va="center",
                color=C["text_light"], fontsize=9, transform=ax.transAxes)
        return

    n_cols = len(col_labels)
    if col_widths is None:
        col_widths = [1 / n_cols] * n_cols

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(C["border"])
        cell.set_linewidth(0.4)
        cell.PAD = 0.07
        if r == 0:
            cell.set_facecolor(C["header_bg"])
            cell.set_text_props(color=C["header_text"], fontweight="bold")
        else:
            is_user = highlight_rows and r in highlight_rows
            if is_user:
                cell.set_facecolor(C["user_bg"])
                cell.set_text_props(color=C["user_text"], fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor(C["row_even"])
                cell.set_text_props(color=C["text"])
            else:
                cell.set_facecolor(C["row_odd"])
                cell.set_text_props(color=C["text"])
        # Aligner le nom à gauche (col 1 sur les pages épreuves, col 0 résumé)
        if c == 1 and r > 0:
            cell.get_text().set_ha("left")


# ── Page résumé ───────────────────────────────────────────────────────────────

def create_summary_page(summary_rows, nom, categorie, date_str):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(C["bg"])

    subtitle = "  ·  ".join(filter(None, [nom, categorie, date_str]))
    add_banner(fig, DOC_TITLE, subtitle)
    add_footer(fig, f"Source : {BASE_URL}   ·   Généré le {date_str}")

    # --- Titre de section ---
    fig.text(0.06, 0.915, "Récapitulatif", fontsize=10, fontweight="bold",
             color=C["banner_bg"], va="top")

    # --- Table résumé (gauche) ---
    ax_tab = fig.add_axes([0.04, 0.04, 0.43, 0.865])
    t_rows, highlights = [], set()
    for i, (epr, cat_rank, cat_total, resultat, is_nom) in enumerate(summary_rows, 1):
        rank_str = f"{cat_rank}/{cat_total}" if cat_rank != "-" else f"-/{cat_total}"
        pct_str  = f"{(cat_total - cat_rank) / cat_total * 100:.0f}%" if cat_rank != "-" else "-"
        t_rows.append([epr, rank_str, pct_str, resultat])
        if is_nom:
            highlights.add(i)
    draw_table(ax_tab,
               col_labels=["Épreuve", "Classement", "%", "Résultat"],
               rows=t_rows,
               highlight_rows=highlights,
               col_widths=[0.36, 0.22, 0.16, 0.26],
               fontsize=8)

    # --- Jauge horizontale (droite) ---
    ax_g = fig.add_axes([0.54, 0.08, 0.43, 0.825])
    _draw_gauge(ax_g, summary_rows, nom)

    return fig


def _draw_gauge(ax, summary_rows, nom):
    # Calcul des valeurs puis tri par % croissant (moins bon → meilleur, bas → haut)
    data = []
    for epr, cat_rank, cat_total, _, _ in summary_rows:
        if cat_rank != "-":
            b = cat_total - cat_rank
            r = cat_rank
            pct = b / cat_total
        else:
            b, r, pct = 0, 0, 0.0
        data.append((epr, b, r, pct))
    data.sort(key=lambda x: x[3])  # croissant : moins bon en haut, meilleur en bas

    labels   = [d[0] for d in data]
    battus_v = [d[1] for d in data]
    rang_v   = [d[2] for d in data]

    y = range(len(labels))
    ax.barh(list(y), battus_v, height=0.55, color=C["battus"], label="Battus")
    ax.barh(list(y), rang_v, left=battus_v, height=0.55, color=C["rang"], label="Rang")

    # Pourcentage affiché à droite de la barre : (participants battus) / total
    # 100% = rang 1 (meilleur), 0% = dernier
    max_total = max((b + r) for b, r in zip(battus_v, rang_v)) or 1
    for i, (b, r) in enumerate(zip(battus_v, rang_v)):
        total = b + r
        if total > 0:
            pct = b / total * 100
            ax.text(total + max_total * 0.03, i, f"{pct:.0f}%",
                    ha="left", va="center", fontsize=8.5, fontweight="bold",
                    color=C["banner_bg"])

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Participants dans la catégorie", fontsize=8)
    ax.set_title(f"Position de {nom}" if nom else "Classements",
                 fontsize=10, fontweight="bold", color=C["banner_bg"], pad=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C["border"])
    ax.spines["bottom"].set_color(C["border"])
    ax.set_facecolor(C["bg"])
    ax.grid(True, axis="x", color=C["grid"], linewidth=0.5, alpha=0.8)
    ax.grid(False, axis="y")
    legend_elements = [
        mpatches.Patch(facecolor=C["battus"], label="Participants battus"),
        mpatches.Patch(facecolor=C["rang"],   label="Votre rang"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=7.5, framealpha=0.85, edgecolor=C["border"])


# ── Page par épreuve ──────────────────────────────────────────────────────────

def create_event_page(label, rows, nom_lower, categorie, date_str,
                      event_analysis=None, priority_rank=None):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(C["bg"])

    event_type = classify_event(label)

    # Trouver le rang utilisateur
    user_rank, user_result = None, None
    for i, row in enumerate(rows, 1):
        if nom_lower and row[1].lower() == nom_lower:
            user_rank, user_result = i, row[3]
            break

    # Bandeau
    pct = 0.0
    if user_rank:
        pct = (len(rows) - user_rank) / len(rows) * 100
        sub = f"{nom_lower.title()}   ·   {user_rank}/{len(rows)}   ·   {pct:.0f}%   ·   {user_result}"
    else:
        sub = categorie or ""
    add_banner(fig, f"Épreuve : {label}", sub)
    add_footer(fig, f"Catégorie : {categorie or 'toutes'}   ·   {date_str}   ·   {BASE_URL}")

    # Titre H1 visible dans le corps de la page
    fig.text(0.5, 0.920, label, fontsize=22, fontweight="bold",
             color=C["banner_bg"], ha="center", va="top")
    fig.text(0.06, 0.893, f"{len(rows)} participants", fontsize=9,
             color=C["text_light"], va="top")

    # --- Courbe (gauche) : pleine hauteur ---
    ax_curve = fig.add_axes([0.06, 0.08, 0.52, 0.790])
    _draw_curve(ax_curve, rows, nom_lower, label, event_type)

    # --- Carte d'info + Tableau (colonne droite) ---
    card_h  = 0.160   # hauteur de la carte d'info
    card_gap = 0.010   # espace entre carte et tableau
    card_y  = 0.870 - card_h          # sommet de la carte
    tab_top = card_y - card_gap       # bas de la carte → haut du tableau
    tab_bot = 0.080
    ax_tab  = fig.add_axes([0.63, tab_bot, 0.35, tab_top - tab_bot])
    _draw_event_table(ax_tab, rows, nom_lower, user_rank, label)

    if user_rank:
        gap_str         = _format_gap_to_first(rows, user_rank, event_type)
        improvement_pct = event_analysis["improvement_pct"] if event_analysis else None
        _draw_info_card(
            fig,
            x0=0.63, card_y=card_y, w=0.35, card_h=card_h,
            user_rank=user_rank, total=len(rows), pct=pct,
            gap_str=gap_str,
            improvement_pct=improvement_pct,
            priority_rank=priority_rank,
        )

    return fig


def _draw_curve(ax, rows, nom_lower, label, event_type):
    x = list(range(1, len(rows) + 1))
    y = [parse_result(r[3]) for r in rows]
    valid = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
    if not valid:
        ax.text(0.5, 0.5, "Résultats non numériques", ha="center", va="center",
                color=C["text_light"], transform=ax.transAxes)
        return

    # ── Filtre outliers ──────────────────────────────────────────────────────
    # Cutoff = user_result ± gap×CHART_OUTLIER_GAP_FACTOR, gap = |user - leader|
    n_filtered = 0
    if nom_lower and CHART_OUTLIER_GAP_FACTOR is not None and event_type is not None:
        user_val = next(
            (parse_result(row[3]) for row in rows if row[1].lower() == nom_lower),
            None,
        )
        if user_val is not None:
            is_dist = (event_type == "distance")
            all_y   = [yi for _, yi in valid]
            best    = max(all_y) if is_dist else min(all_y)
            gap     = abs(user_val - best)
            if gap > 0:
                margin = gap * CHART_OUTLIER_GAP_FACTOR
                if is_dist:
                    cutoff = user_val - margin    # exclure < cutoff
                    filtered = [(xi, yi) for xi, yi in valid if yi >= cutoff]
                else:
                    cutoff = user_val + margin    # exclure > cutoff
                    filtered = [(xi, yi) for xi, yi in valid if yi <= cutoff]
                n_filtered = len(valid) - len(filtered)
                if n_filtered > 0:
                    valid = filtered
    # ────────────────────────────────────────────────────────────────────────

    xs, ys = zip(*valid)

    # Ligne + tous les points
    ax.plot(xs, ys, color=C["curve"], linewidth=1.2, alpha=0.45, zorder=2)
    ax.scatter(xs, ys, color=C["dot"], s=22, alpha=0.65, zorder=3, label="Participants")

    # Point rouge utilisateur
    if nom_lower:
        for i, row in enumerate(rows, 1):
            if row[1].lower() == nom_lower:
                uy = parse_result(row[3])
                if uy is not None:
                    ax.scatter([i], [uy], color=C["user_dot"], s=90, zorder=5,
                               edgecolors="white", linewidths=1.2, label="Vous")
                    ax.axvline(i, color=C["user_dot"], linestyle="--",
                               alpha=0.35, linewidth=1)
                    # Annotation avec décalage intelligent
                    y_range = max(ys) - min(ys) if max(ys) != min(ys) else 1
                    x_frac = i / len(rows)
                    xytext = (-55, 6) if x_frac > 0.75 else (8, 6)
                    ax.annotate(
                        f" #{i} — {row[3]} ",
                        (i, uy), xytext=xytext,
                        textcoords="offset points",
                        fontsize=8, fontweight="bold", color=C["user_dot"],
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec=C["user_dot"], alpha=0.9, lw=0.8),
                        arrowprops=dict(arrowstyle="-", color=C["user_dot"],
                                        lw=0.8, alpha=0.7),
                    )

    ylabel = "Distance (m)" if event_type == "distance" else "Temps (s)"
    if event_type == "time":
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: (
                f"{int(v//60)}:{v%60:05.2f}" if v >= 60 else f"{v:.1f}s"
            ))
        )

    ax.set_xlabel("Classement dans la catégorie", fontsize=9, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=9, labelpad=6)
    ax.set_xlim(0, len(rows) + 1)
    ax.spines["left"].set_color(C["border"])
    ax.spines["bottom"].set_color(C["border"])
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, framealpha=0.85, edgecolor=C["border"],
              loc="upper right" if event_type == "distance" else "lower right")

    # Axe Y droit : allure au 500m
    def _fmt_pace_ax(s, _=None):
        if s <= 0:
            return ""
        return f"{int(s // 60)}:{s % 60:04.1f}"

    if label in DIST_EV_PACE:
        d = DIST_EV_PACE[label]
        ax2 = ax.secondary_yaxis(
            "right",
            functions=(lambda y, d=d: y / d * 500, lambda p, d=d: p * d / 500),
        )
        ax2.set_ylabel("Allure /500m", fontsize=8, color=C["text_light"], labelpad=6)
        ax2.yaxis.set_major_formatter(FuncFormatter(_fmt_pace_ax))
        ax2.tick_params(labelsize=7.5, colors=C["text_light"])
        ax.spines["right"].set_visible(False)

    elif label in TIME_EV_PACE:
        dur = TIME_EV_PACE[label]
        ax2 = ax.secondary_yaxis(
            "right",
            functions=(
                lambda y, d=dur: np.where(np.asarray(y) > 0, d / np.asarray(y) * 500, np.nan),
                lambda p, d=dur: np.where(np.asarray(p) > 0, d * 500 / np.asarray(p), np.nan),
            ),
        )
        ax2.set_ylabel("Allure /500m", fontsize=8, color=C["text_light"], labelpad=6)
        ax2.yaxis.set_major_formatter(FuncFormatter(_fmt_pace_ax))
        ax2.tick_params(labelsize=7.5, colors=C["text_light"])
        ax.spines["right"].set_visible(False)

    if n_filtered > 0:
        ax.text(0.98, 0.02,
                f"{n_filtered} participant(s) hors échelle masqué(s)",
                transform=ax.transAxes, fontsize=6.5, color=C["text_light"],
                ha="right", va="bottom", style="italic")


def _draw_event_table(ax, rows, nom_lower, user_rank, label):
    """
    Affiche :
      - Les 3 premiers (podium)
      - Un séparateur "…" si écart
      - Les 14 lignes juste avant le rang de l'utilisateur
      - La ligne de l'utilisateur (surlignée)
      - Les 5 lignes suivant l'utilisateur (sauf s'il est dernier)
    """
    def pace_str(result_raw):
        val = parse_result(result_raw)
        if not val:
            return "–"
        if label in DIST_EV_PACE:
            p = val / DIST_EV_PACE[label] * 500
        elif label in TIME_EV_PACE:
            p = TIME_EV_PACE[label] / val * 500 if val > 0 else None
            if p is None:
                return "–"
        else:
            return "–"
        return f"{int(p // 60)}:{p % 60:04.1f}"

    def fmt(row, idx):
        name = row[1][:17] + "…" if len(row[1]) > 17 else row[1]
        return [f"#{idx}", name, row[3], pace_str(row[3])]

    t_rows, highlights = [], set()

    if not user_rank:
        # Pas de personne cible : top 25
        for idx, row in enumerate(rows[:25], 1):
            t_rows.append(fmt(row, idx))
    else:
        # Bloc 1 : top 3
        top3_end = min(3, user_rank)
        for idx in range(1, top3_end + 1):
            t_rows.append(fmt(rows[idx - 1], idx))
            if nom_lower and rows[idx - 1][1].lower() == nom_lower:
                highlights.add(len(t_rows))

        if user_rank <= 3:
            pass  # l'utilisateur est déjà dans le top 3
        else:
            # Bloc 2 : 14 lignes avant l'utilisateur (inclus)
            window_start = max(4, user_rank - 13)  # 14 lignes max, sans chevaucher top 3

            if window_start > 4:
                t_rows.append(["…", f"({window_start - 4} lignes)", "…", ""])

            for idx in range(window_start, user_rank + 1):
                t_rows.append(fmt(rows[idx - 1], idx))
                if nom_lower and rows[idx - 1][1].lower() == nom_lower:
                    highlights.add(len(t_rows))

        # Bloc 3 : 5 lignes après l'utilisateur (sauf s'il est dernier)
        if user_rank < len(rows):
            after_end = min(user_rank + 5, len(rows))
            for idx in range(user_rank + 1, after_end + 1):
                t_rows.append(fmt(rows[idx - 1], idx))

    draw_table(ax,
               col_labels=["#", "Nom", "Résultat", "/500m"],
               rows=t_rows,
               highlight_rows=highlights,
               col_widths=[0.12, 0.38, 0.25, 0.25],
               fontsize=6.5)


# ── Page de garde ────────────────────────────────────────────────────────────

def create_cover_page(nom, categorie, date_str, n_events):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(C["banner_bg"])   # fond bleu marine pleine page

    # Bande décorative claire en haut
    ax_top = fig.add_axes([0.0, 0.88, 1.0, 0.12])
    ax_top.set_facecolor("#142D5A")
    ax_top.axis("off")

    # Trait horizontal accent
    for y_pos, alpha in [(0.875, 0.9), (0.870, 0.4)]:
        ax_line = fig.add_axes([0.08, y_pos, 0.84, 0.003])
        ax_line.set_facecolor(C["rang"])
        ax_line.set_alpha(alpha)
        ax_line.axis("off")

    # Titre principal (multi-lignes)
    fig.text(0.5, 0.74, "Analyseur de performance C7DC",
             ha="center", va="center", fontsize=24, fontweight="bold",
             color="white")
    fig.text(0.5, 0.66, "et axes de progression",
             ha="center", va="center", fontsize=20,
             color=C["banner_sub"])

    # Séparateur
    ax_sep = fig.add_axes([0.20, 0.60, 0.60, 0.002])
    ax_sep.set_facecolor(C["rang"])
    ax_sep.axis("off")

    # Informations participant
    if nom:
        fig.text(0.5, 0.54, nom.upper(),
                 ha="center", va="center", fontsize=17, fontweight="bold",
                 color="white")
    if categorie:
        fig.text(0.5, 0.47, f"Catégorie : {categorie}",
                 ha="center", va="center", fontsize=13,
                 color=C["banner_sub"])

    fig.text(0.5, 0.40, f"{n_events} épreuve(s) analysée(s)",
             ha="center", va="center", fontsize=11, color="#7A9ED4")

    # Date
    fig.text(0.5, 0.32, f"Généré le {date_str}",
             ha="center", va="center", fontsize=10, color="#5A80B0")

    # Pied de page discret
    fig.text(0.5, 0.07, BASE_URL,
             ha="center", va="center", fontsize=8, color="#3A5F8A")
    fig.text(0.5, 0.04, "C7DC — 7 Défis Capitaux",
             ha="center", va="center", fontsize=8, color="#3A5F8A")
    fig.text(0.5, 0.015, "github.com/alemorvan/crap-7dc  ·  Licence CC BY-SA 4.0",
             ha="center", va="center", fontsize=7, color="#8AAACC")

    return fig


# ── Page 1/5 : Tableau des opportunités ──────────────────────────────────────

def create_opportunities_page(analysis, nom, categorie, date_str):
    """Page 1 : tableau des opportunités de progression."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(C["bg"])
    subtitle = "  ·  ".join(filter(None, [nom, categorie, date_str]))
    add_banner(fig, "Tableau des opportunités de progression", subtitle)
    add_footer(fig, f"Source : {BASE_URL}   ·   Généré le {date_str}")

    if not analysis:
        fig.text(0.5, 0.5, "Données insuffisantes.", ha="center", va="center",
                 fontsize=12, color=C["text_light"])
        return fig

    total_points = sum(a["current_points"] for a in analysis)

    fig.text(0.04, 0.905, "Classement des opportunités de progression",
             fontsize=11, fontweight="bold", color=C["banner_bg"], va="bottom")
    fig.text(0.04, 0.893, f"Total : {total_points} points bonus — {len(analysis)} épreuve(s) analysée(s)",
             fontsize=9, color=C["text_light"], va="top")

    ax_tab = fig.add_axes([0.04, 0.55, 0.92, 0.33])
    opp_rows = []
    for rank_i, a in enumerate(analysis, 1):
        priority = {1: "1re priorité", 2: "2e priorité", 3: "3e priorité"}.get(rank_i, "")
        opp_rows.append([
            a["label"],
            f"{a['current_pct']:.0f}%",
            f"{a['current_points']} pts",
            str(a["density_5pct"]),
            f"{a['improvement_pct']:.1f}%",
            priority,
        ])
    highlights = {1, 2} if len(analysis) > 1 else {1}
    draw_table(
        ax_tab,
        col_labels=["Épreuve", "Score %", "Points", "Conc. à <5%", "Effort +10 places", "Priorité"],
        rows=opp_rows,
        highlight_rows=highlights,
        col_widths=[0.17, 0.11, 0.12, 0.15, 0.20, 0.25],
        fontsize=8,
    )

    # Légende explicative
    y = 0.535
    legends = [
        ("Score %",          "Ton pourcentage relatif dans la catégorie (100% = meilleur)."),
        ("Points",           "Points bonus actuels (total - rang)."),
        ("Conc. à <5%",      "Nombre de concurrents dans les 5% qui te précèdent directement."),
        ("Effort +10 places","Amélioration de résultat nécessaire pour gagner 10 places."),
        ("Priorité",         "Classement de rentabilité : densité × facilité de progression."),
    ]
    for label_l, desc in legends:
        if y < 0.04:
            break
        fig.text(0.04, y, f"• {label_l} :", fontsize=8, fontweight="bold",
                 color=C["banner_bg"], va="top")
        fig.text(0.20, y, desc, fontsize=8, color=C["text"], va="top")
        y -= 0.022

    return fig


# ── Page 2/5 : Préconisations personnalisées ─────────────────────────────────

def create_recommendations_page(analysis, nom, categorie, date_str):
    """Page 2 : préconisations textuelles rule-based."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(C["bg"])
    subtitle = "  ·  ".join(filter(None, [nom, categorie, date_str]))
    add_banner(fig, "Préconisations personnalisées", subtitle)
    add_footer(fig, f"Source : {BASE_URL}   ·   Généré le {date_str}")

    if not analysis:
        fig.text(0.5, 0.5, "Données insuffisantes.", ha="center", va="center",
                 fontsize=12, color=C["text_light"])
        return fig

    total_points = sum(a["current_points"] for a in analysis)
    recs = _build_recs(analysis, total_points)

    fig.text(0.04, 0.905, "Préconisations personnalisées",
             fontsize=11, fontweight="bold", color=C["banner_bg"], va="bottom")

    y = 0.882
    for title, body in recs:
        if y < 0.04:
            break
        fig.text(MARGIN_L, y, f"• {title}", fontsize=9.5, fontweight="bold",
                 color=C["banner_bg"], va="top")
        y -= 0.028
        for line in textwrap.wrap(body, width=WRAP[8.5]):
            if y < 0.04:
                break
            fig.text(MARGIN_L + 0.012, y, line, fontsize=8.5, color=C["text"], va="top")
            y -= 0.020
        y -= 0.012

    return fig


# ── Page profil athlétique ────────────────────────────────────────────────────

def _compute_zones(analysis):
    """Score moyen par zone énergétique (uniquement les zones présentes dans l'analyse)."""
    zones = {}
    for key, events, label in ENERGY_ZONES:
        matching = [a for a in analysis if a["label"] in events]
        if matching:
            zones[key] = {
                "avg":    sum(a["current_pct"] for a in matching) / len(matching),
                "label":  label,
                "events": matching,
            }
    return zones


def _detect_profile(zones):
    """
    Retourne (nom_profil, description, [(titre_reco, corps_reco), ...]).
    Détecte 4 profils : V (seuil creux), explosif, endurant, polyvalent.
    """
    sc = {k: v["avg"] for k, v in zones.items()}

    # Score "haut" : meilleur des zones courtes/VO2max
    peak = max(sc.get("Sprint", 0), sc.get("Puissance", 0), sc.get("VO2max", 0))
    seuil      = sc.get("Seuil", 0)
    endurance  = sc.get("Endurance", 0)
    short_avg  = sum(sc[k] for k in ("Sprint", "Puissance") if k in sc) / max(1, sum(1 for k in ("Sprint", "Puissance") if k in sc))

    # ── Profil en V : fort en puissance/VO2max + correct en endurance, creux au seuil ──
    if "Seuil" in sc and "Endurance" in sc:
        if peak > seuil + 10 and endurance > seuil + 5:
            return (
                "Profil en 'V' — seuil lactique sous-développé",
                "Tu es fort sur les efforts courts (puissance/VO2max) et correct sur l'endurance longue, "
                "mais tu chutes nettement sur les distances intermédiaires (2000 m, 5000 m, 30 min). "
                "Ce 'V' est typique d'un athlète qui s'entraîne aux extrêmes — sprints courts "
                "ou longues sorties — sans passer par la zone d'inconfort du seuil lactique (6–20 min "
                "d'effort soutenu). C'est le maillon manquant.",
                [
                    ("1. Endurance de force — priorité absolue",
                     "Intervalles longs au seuil : 4–5 × 6–8 min à allure soutenue mais tenable, récupération 2 min. "
                     "Cible directe : 2000 m, 5000 m, 30 min. C'est le travail avec le meilleur retour sur investissement."),
                    ("2. Endurance fondamentale",
                     "Sorties continues 30–60 min en zone 2 (allure où tu peux tenir une conversation). "
                     "Consolide la base aérobie et améliore les scores sur 30 min et 10 000 m."),
                    ("3. Musculation — faible priorité",
                     "Déjà fort sur les épreuves de puissance courte. Quelques exercices de force spécifique "
                     "(tirage lourd, squats) peuvent légèrement aider le 100 m, "
                     "mais le retour est limité comparé au travail de seuil."),
                ]
            )

    # ── Profil explosif : fort court, faible long ──
    if "Endurance" in sc and short_avg > endurance + 15:
        return (
            "Profil explosif — base aérobie à développer",
            "Tes épreuves courtes sont clairement tes points forts, mais la filière aérobie longue est sous-développée. "
            "Ton moteur est puissant mais il n'a pas l'endurance pour le faire tourner longtemps.",
            [
                ("1. Endurance fondamentale",
                 "Sorties longues régulières en zone 2 (30–60 min, faible intensité). "
                 "Indispensable pour élever le plafond aérobie et améliorer 10 000 m et 60 min."),
                ("2. Endurance de force",
                 "Intervalles de 5–8 min au seuil pour faire le pont entre puissance et endurance. "
                 "Améliore 5000 m et 30 min."),
                ("3. Maintenir la puissance",
                 "Conserver les séances explosives courtes pour ne pas perdre les qualités acquises."),
            ]
        )

    # ── Profil endurant : fort long, faible court ──
    if "Endurance" in sc and endurance > short_avg + 15:
        return (
            "Profil endurant — puissance explosive à développer",
            "Ton endurance est ta signature : tu excelles sur les longues distances mais la puissance "
            "explosive te fait défaut. Ton moteur tourne longtemps mais manque de cylindrée sur les efforts courts.",
            [
                ("1. Intervalles courts explosifs",
                 "8 × 1 min à intensité maximale, récupération 2 min. "
                 "Cibles directes : 100 m, 500 m, 1 min."),
                ("2. Musculation / force spécifique",
                 "Renforcement des jambes et du dos (squats, tirage horizontal) "
                 "pour augmenter la force appliquée sur l'ergomètre."),
                ("3. Endurance de force",
                 "Intervalles de 5–6 min à allure tempo pour créer le lien entre ta base aérobie et la puissance."),
            ]
        )

    # ── Profil polyvalent ──
    return (
        "Profil polyvalent — progression ciblée",
        "Ton niveau est homogène sur toutes les filières énergétiques : pas de lacune structurelle majeure. "
        "La stratégie la plus efficace est de concentrer l'entraînement sur les épreuves "
        "à fort potentiel de gain identifiées dans le tableau des opportunités.",
        [
            ("1. Travailler les épreuves prioritaires",
             "Concentre-toi sur les 2–3 premières épreuves du tableau des opportunités (page précédente). "
             "Ce sont les plus rentables en termes de points bonus."),
            ("2. Séances variées",
             "Alterner endurance, intervalles et puissance pour maintenir l'équilibre acquis."),
            ("3. Récupération et régularité",
             "À ce niveau d'homogénéité, la qualité de récupération (sommeil, nutrition) "
             "et la régularité de l'entraînement sont souvent les facteurs limitants."),
        ]
    )


def _draw_zone_bars(ax, zones, analysis):
    """Barres horizontales colorées par zone énergétique, du sprint à l'endurance."""
    order   = ["Sprint", "Puissance", "VO2max", "Seuil", "Endurance"]
    present = [(k, zones[k]) for k in order if k in zones]
    if not present:
        return

    labels  = [v["label"] for _, v in present]
    avgs    = [v["avg"]   for _, v in present]
    overall = sum(avgs) / len(avgs)

    def bar_color(pct):
        if pct >= 60:  return C["battus"]   # vert — bon
        if pct >= 40:  return C["rang"]     # bleu — moyen
        return "#D62728"                     # rouge — faible

    colors = [bar_color(a) for a in avgs]
    y = list(range(len(labels)))

    ax.barh(y, avgs, height=0.55, color=colors)

    for i, avg in enumerate(avgs):
        x_text = max(avg - 2, 2)
        ax.text(x_text, i, f"{avg:.0f}%",
                ha="right", va="center", fontsize=9, fontweight="bold", color="white")

    ax.axvline(overall, color=C["banner_bg"], linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"Moyenne ({overall:.0f}%)")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score % (100 % = meilleur)", fontsize=8)
    ax.set_title("Score par filière énergétique", fontsize=9.5, fontweight="bold",
                 color=C["banner_bg"], pad=6)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C["border"])
    ax.spines["bottom"].set_color(C["border"])
    ax.set_facecolor(C["bg"])
    ax.grid(True, axis="x", color=C["grid"], linewidth=0.5, alpha=0.8)
    ax.grid(False, axis="y")
    ax.legend(fontsize=7.5, framealpha=0.85, edgecolor=C["border"], loc="lower right")


def create_profile_page(analysis, nom, categorie, date_str, ai_analysis=None):
    """
    Page de profil athlétique : graphique par filière + diagnostic + plan d'entraînement.
    Si ai_analysis est fourni (dict issu de l'API Claude), son texte remplace le rule-based.
    """
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(C["bg"])

    subtitle = "  ·  ".join(filter(None, [nom, categorie, date_str]))
    src = "Analyse IA (Claude)" if ai_analysis else "Analyse automatique"
    add_banner(fig, f"Profil athlétique et plan d'entraînement  [{src}]", subtitle)
    add_footer(fig, f"Source : {BASE_URL}   ·   Généré le {date_str}")

    if not analysis:
        fig.text(0.5, 0.5, "Données insuffisantes.", ha="center", va="center",
                 fontsize=12, color=C["text_light"])
        return fig

    zones = _compute_zones(analysis)
    if not zones:
        return fig

    # ── Résolution des textes (IA ou rule-based) ──
    if ai_analysis:
        profile_name  = ai_analysis.get("profil_nom", "Profil inconnu")
        profile_desc  = ai_analysis.get("profil_description", "")
        training_recs = [
            (item["titre"], item["description"])
            for item in ai_analysis.get("plan_entrainement", [])
        ]
        extra_sections = [
            ("Épreuves cibles",  ai_analysis.get("epreuves_cibles", "")),
            ("Conclusion",       ai_analysis.get("conclusion", "")),
        ]
    else:
        profile_name, profile_desc, training_recs = _detect_profile(zones)
        extra_sections = []

    WRAP_W = 120

    # ── Page 3/5 : Filières + Diagnostic ────────────────────────────────────
    # Zone bars centrée — marge gauche accrue pour les labels des axes y
    ax_zones = fig.add_axes([0.22, 0.610, 0.73, 0.270])
    _draw_zone_bars(ax_zones, zones, analysis)

    # Nom du profil en bandeau coloré juste sous le graphique
    ax_pname = fig.add_axes([MARGIN_L, 0.575, MARGIN_R - MARGIN_L, 0.030])
    ax_pname.set_facecolor(C["banner_bg"])
    ax_pname.axis("off")
    ax_pname.text(0.5, 0.5, profile_name, ha="center", va="center",
                  fontsize=9.5, fontweight="bold", color="white",
                  transform=ax_pname.transAxes)

    # Diagnostic sous le profil, avec espacement clair
    fig.text(MARGIN_L, 0.550, "Diagnostic", fontsize=11, fontweight="bold",
             color=C["banner_bg"], va="bottom")
    y = 0.532
    for line in textwrap.wrap(profile_desc, width=WRAP[9]):
        if y < 0.04:
            break
        fig.text(MARGIN_L, y, line, fontsize=9, color=C["text"], va="top")
        y -= 0.021

    return fig


def create_plan_page(analysis, nom, categorie, date_str, ai_analysis=None):
    """Page 4/5 : plan d'action + épreuves cibles."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(C["bg"])
    subtitle = "  ·  ".join(filter(None, [nom, categorie, date_str]))
    src = "Analyse IA (Claude)" if ai_analysis else "Analyse automatique"
    add_banner(fig, f"Plan d'action  [{src}]", subtitle)
    add_footer(fig, f"Source : {BASE_URL}   ·   Généré le {date_str}")

    zones = _compute_zones(analysis)

    if ai_analysis:
        training_recs = [(item["titre"], item["description"])
                         for item in ai_analysis.get("plan_entrainement", [])]
        epreuves_cibles = ai_analysis.get("epreuves_cibles", "")
    else:
        _, _, training_recs = _detect_profile(zones)
        epreuves_cibles = ""

    WRAP_W = 120

    fig.text(MARGIN_L, 0.905, "Plan d'entraînement recommandé", fontsize=11,
             fontweight="bold", color=C["banner_bg"], va="bottom")
    y = 0.882

    for title, body in training_recs:
        if y < 0.04:
            break
        fig.text(MARGIN_L, y, f"• {title}", fontsize=9.5, fontweight="bold",
                 color=C["banner_bg"], va="top")
        y -= 0.025
        for line in textwrap.wrap(body, width=WRAP[8.5]):
            if y < 0.04:
                break
            fig.text(MARGIN_L + 0.012, y, line, fontsize=8.5, color=C["text"], va="top")
            y -= 0.020
        y -= 0.012

    if epreuves_cibles and y > 0.12:
        y -= 0.010
        fig.text(MARGIN_L, y, "Épreuves cibles", fontsize=11, fontweight="bold",
                 color=C["banner_bg"], va="bottom")
        y -= 0.025
        for line in textwrap.wrap(epreuves_cibles, width=WRAP[8.5]):
            if y < 0.04:
                break
            fig.text(MARGIN_L, y, line, fontsize=8.5, color=C["text"], va="top")
            y -= 0.020

    return fig


def create_conclusion_page(nom, categorie, date_str, ai_analysis=None):
    """Page 5/5 : conclusion."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(C["bg"])
    subtitle = "  ·  ".join(filter(None, [nom, categorie, date_str]))
    add_banner(fig, "Conclusion", subtitle)
    add_footer(fig, f"Source : {BASE_URL}   ·   Généré le {date_str}")

    conclusion = ai_analysis.get("conclusion", "") if ai_analysis else ""

    if not conclusion:
        fig.text(0.5, 0.5, "Aucune conclusion disponible.\nAjoutez un fichier d'analyse IA.",
                 ha="center", va="center", fontsize=11, color=C["text_light"],
                 multialignment="center")
        return fig

    fig.text(MARGIN_L, 0.905, "Conclusion", fontsize=11, fontweight="bold",
             color=C["banner_bg"], va="bottom")

    # Fond légèrement coloré — marges 1cm de chaque côté
    ax_bg = fig.add_axes([MARGIN_L, 0.085, MARGIN_R - MARGIN_L, 0.800])
    ax_bg.set_facecolor("#EEF3FA")
    ax_bg.axis("off")

    y = 0.870
    for line in textwrap.wrap(conclusion, width=WRAP[9]):
        fig.text(MARGIN_L + 0.012, y, line, fontsize=9, color=C["text"], va="top")
        y -= 0.022

    # Trait décoratif et source
    ax_line = fig.add_axes([MARGIN_L, 0.095, MARGIN_R - MARGIN_L, 0.002])
    ax_line.set_facecolor(C["banner_bg"])
    ax_line.axis("off")
    fig.text(0.5, 0.085, "Analyse générée par Claude (Anthropic) à partir des données C7DC",
             ha="center", va="top", fontsize=8, color=C["text_light"], style="italic")

    return fig


# ── Courbe d'allure individuelle ──────────────────────────────────────────────

DIST_EV_PACE = {"100 m": 100, "500 m": 500, "1000 m": 1000,
                "2000 m": 2000, "5000 m": 5000, "10000 m": 10000}
TIME_EV_PACE = {"1 min": 60, "4 min": 240, "30 min": 1800, "60 min": 3600}


def create_pace_curve_page(event_data, nom, nom_lower, categorie, date_str):
    """Page : courbe d'allure au 500m par épreuve pour un participant."""
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D

    # Calcul des allures
    points = []
    for label, rows in event_data.items():
        user_result = None
        for row in rows:
            if nom_lower and row[1].lower() == nom_lower:
                user_result = row[3]
                break
        if user_result is None:
            continue
        val = parse_result(user_result)
        if not val:
            continue
        if label in DIST_EV_PACE:
            dist_m = DIST_EV_PACE[label]
            pace_s = val / dist_m * 500
            etype = "dist"
        elif label in TIME_EV_PACE:
            dist_m = val
            pace_s = TIME_EV_PACE[label] / dist_m * 500
            etype = "time"
        else:
            continue
        points.append((dist_m, pace_s, label, etype))

    points.sort(key=lambda p: p[0])

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(C["bg"])
    subtitle = "  ·  ".join(filter(None, [nom, categorie, date_str]))
    add_banner(fig, "Courbe d'allure au 500 m", subtitle)
    add_footer(fig, f"Source : {BASE_URL}   ·   Généré le {date_str}")

    fig.text(MARGIN_L, 0.920, "Profil de vitesse par épreuve",
             fontsize=11, fontweight="bold", color=C["banner_bg"], va="top")

    ax = fig.add_axes([0.10, 0.130, 0.86, 0.745])
    ax.set_facecolor(C["bg"])

    if not points:
        ax.text(0.5, 0.5, "Aucune donnée disponible",
                ha="center", va="center", fontsize=12, color=C["text_light"],
                transform=ax.transAxes)
        return fig

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    # Ligne de connexion
    ax.plot(xs, ys, color=C["rang"], linewidth=1.8, alpha=0.55, zorder=2, linestyle="--")

    # Zone de fond entre la courbe et le bas du graphe
    ax.fill_between(xs, ys, max(ys) * 1.05, alpha=0.07, color=C["rang"], zorder=1)

    # Points + annotations
    offsets_y = []
    for i, (x, y, lbl, et) in enumerate(points):
        color = C["user_dot"] if et == "dist" else C["banner_bg"]
        marker = "o" if et == "dist" else "D"
        ax.scatter([x], [y], color=color, s=70, zorder=5,
                   marker=marker, edgecolors="white", linewidths=1.2)

        pace_m, pace_s = int(y // 60), y % 60
        pace_str = f"{pace_m}:{pace_s:04.1f}"

        # Alternance haut/bas pour éviter les chevauchements
        vert_offset = 14 if i % 2 == 0 else -22
        va = "bottom" if vert_offset > 0 else "top"

        ax.annotate(
            f"{lbl}\n{pace_str}",
            (x, y),
            xytext=(0, vert_offset),
            textcoords="offset points",
            ha="center", va=va,
            fontsize=7.5, color=C["text"],
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=C["border"], alpha=0.88, lw=0.6),
            zorder=6,
        )

    # Axes log + inverted
    ax.set_xscale("log")
    ax.invert_yaxis()

    # X ticks sur les distances standard
    dist_ticks = sorted(DIST_EV_PACE.values())
    ax.set_xticks(dist_ticks)
    ax.set_xticklabels(
        [k for d in dist_ticks for k, v in DIST_EV_PACE.items() if v == d],
        fontsize=8.5,
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    # Y ticks min:sec
    def fmt_pace(s, _=None):
        return f"{int(s // 60)}:{s % 60:04.1f}"

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_pace))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))

    ax.set_xlabel("Distance parcourue", fontsize=9, labelpad=8, color=C["text"])
    ax.set_ylabel("Allure au 500 m  (min:sec)", fontsize=9, labelpad=8, color=C["text"])

    ax.grid(True, axis="y", color=C["grid"], linewidth=0.6, alpha=0.9)
    ax.grid(True, axis="x", color=C["grid"], linewidth=0.4, alpha=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(C["border"])
    ax.spines["bottom"].set_color(C["border"])
    ax.tick_params(labelsize=8.5, colors=C["text"])

    # Légende
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C["user_dot"],
               markeredgecolor="white", markeredgewidth=0.8, markersize=8,
               label="Épreuve distance fixe"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=C["banner_bg"],
               markeredgecolor="white", markeredgewidth=0.8, markersize=8,
               label="Épreuve durée fixe (dist. variable)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower left",
              framealpha=0.92, edgecolor=C["border"])

    # Note explicative
    fig.text(0.5, 0.108, "Plus haut = plus rapide   ·   Abscisse en échelle logarithmique",
             ha="center", va="top", fontsize=7.5, color=C["text_light"], style="italic")

    return fig


# ── Génération du PDF ─────────────────────────────────────────────────────────

def generate_pdf(output_file, summary_rows, event_data, nom, nom_lower, categorie,
                 ai_analysis=None):
    setup_style()
    date_str = datetime.now().strftime("%d/%m/%Y")

    # Tri EVENT_ORDER (épreuves + résumé)
    event_data    = _sort_events(event_data)
    summary_rows  = sorted(
        summary_rows,
        key=lambda r: (0, EVENT_ORDER.index(r[0])) if r[0] in EVENT_ORDER else (1, r[0])
    )

    # Analyse pré-calculée pour alimenter les pages d'épreuves
    analysis          = compute_analysis(event_data, nom_lower) if nom_lower else []
    event_analysis_map = {a["label"]: a for a in analysis}
    priority_map       = {a["label"]: i for i, a in enumerate(analysis, 1) if i <= 3}

    with PdfPages(output_file) as pdf:
        print("  Page de garde...")
        fig = create_cover_page(nom, categorie, date_str, len(event_data))
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

        print("  Page résumé...")
        fig = create_summary_page(summary_rows, nom, categorie, date_str)
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

        if nom_lower:
            print("  Courbe d'allure au 500 m...")
            fig = create_pace_curve_page(event_data, nom, nom_lower, categorie, date_str)
            pdf.savefig(fig, bbox_inches="tight", dpi=150)
            plt.close(fig)

        for label, rows in event_data.items():
            print(f"  Page : {label} ({len(rows)} participants)...")
            fig = create_event_page(
                label, rows, nom_lower, categorie, date_str,
                event_analysis=event_analysis_map.get(label),
                priority_rank=priority_map.get(label),
            )
            pdf.savefig(fig, bbox_inches="tight", dpi=150)
            plt.close(fig)

        if nom_lower:
            print("  [1/5] Tableau des opportunités...")
            fig = create_opportunities_page(analysis, nom, categorie, date_str)
            pdf.savefig(fig, bbox_inches="tight", dpi=150)
            plt.close(fig)

            print("  [2/5] Préconisations personnalisées...")
            fig = create_recommendations_page(analysis, nom, categorie, date_str)
            pdf.savefig(fig, bbox_inches="tight", dpi=150)
            plt.close(fig)


        # Métadonnées PDF
        d = pdf.infodict()
        d["Title"]   = DOC_TITLE
        d["Author"]  = nom or "C7DC Scraper"
        d["Subject"] = f"Classements C7DC — {categorie or 'toutes catégories'}"
        d["Creator"] = "scrape_classement.py"

    n_pages = 2 + len(event_data) + (3 if nom_lower else 0)  # courbe allure + opportunités + préconisations
    print(f"\nFichier PDF généré : {output_file}  ({n_pages} pages)")


# ── Export JSON ──────────────────────────────────────────────────────────────

def save_json(filepath, nom, categorie, no_extra, summary_rows, event_data):
    """Sauvegarde les données scrapées dans un fichier JSON réutilisable."""
    data = {
        "meta": {
            "nom":        nom,
            "categorie":  categorie,
            "no_extra":   no_extra,
            "scraped_at": datetime.now().isoformat(),
        },
        "summary": [
            {
                "label":     label,
                "cat_rank":  cat_rank,
                "cat_total": cat_total,
                "resultat":  resultat,
                "is_nom":    is_nom,
            }
            for label, cat_rank, cat_total, resultat, is_nom in summary_rows
        ],
        "events": {
            label: [list(row) for row in rows]
            for label, rows in event_data.items()
        },
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Données JSON sauvegardées : {filepath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scrape les classements ffaviron et génère un rapport PDF."
    )
    parser.add_argument("--categorie", metavar="CAT", default=None,
                        help='Filtrer par catégorie, ex: "H 40-49 TC"')
    parser.add_argument("--no-extra", action="store_true", default=False,
                        help="Exclure les épreuves longues/saisonnières")
    parser.add_argument("--nom", metavar="NOM", default=None,
                        help='Mettre en avant ce participant, ex: "ANTOINE LE MORVAN"')
    args = parser.parse_args()

    session = get_session()

    print(f"Récupération des épreuves depuis {BASE_URL}...")
    html = fetch_page(session, 1)
    epreuves = get_epreuves(html)

    if not epreuves:
        print("Aucune épreuve trouvée.")
        sys.exit(1)

    epreuves = {
        eid: label for eid, label in epreuves.items()
        if not is_competition(label)
        and (not args.no_extra or label not in EXTRA_LABELS)
    }

    print(f"{len(epreuves)} épreuve(s) : {', '.join(epreuves.values())}")

    nom_lower = args.nom.lower() if args.nom else None

    # Auto-détection de la catégorie quand --nom est fourni sans --categorie
    categorie = args.categorie
    if nom_lower and not categorie:
        print(f"Détection automatique de la catégorie pour '{args.nom}'...")
        categorie = auto_detect_category(session, epreuves, nom_lower)
        if categorie:
            print(f"  → Catégorie détectée : '{categorie}'")
        else:
            print(f"  ⚠ Participant '{args.nom}' introuvable — extraction sans filtre catégorie.")

    if categorie:
        print(f"Catégorie : '{categorie}'")
    if args.nom:
        print(f"Participant mis en avant : '{args.nom}'")

    output_file = build_output_filename(categorie, args.no_extra, args.nom)

    summary_rows, event_data = [], {}

    for i, (eid, label) in enumerate(epreuves.items()):
        print(f"  [{i+1}/{len(epreuves)}] '{label}'...", end="", flush=True)
        try:
            html = fetch_page(session, eid, delay=(i > 0))
        except Exception as e:
            print(f" erreur : {e}")
            continue

        rows = parse_all_tables(html, categorie_filter=categorie)
        if categorie and not rows:
            print(f" aucun résultat pour '{categorie}', ignoré.")
            continue

        cat_rank, nom_resultat = "-", "-"
        for idx, row in enumerate(rows, 1):
            if nom_lower and row[1].lower() == nom_lower:
                cat_rank, nom_resultat = idx, row[3]
                break

        print(f" {len(rows)} participant(s).")
        summary_rows.append((label, cat_rank, len(rows), nom_resultat, cat_rank != "-"))
        event_data[label] = rows

    if not summary_rows:
        print("\nAucun résultat collecté.")
        sys.exit(1)

    json_file = output_file.replace(".pdf", ".json")
    save_json(json_file, args.nom, categorie, args.no_extra, summary_rows, event_data)

    print("\nGénération du PDF...")
    generate_pdf(output_file, summary_rows, event_data, args.nom, nom_lower, categorie)


if __name__ == "__main__":
    main()
