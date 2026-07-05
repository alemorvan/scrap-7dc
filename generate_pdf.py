#!/usr/bin/env python3
"""
Génère un PDF de classement C7DC depuis un fichier JSON (produit par scrape_classement.py).
Détecte automatiquement un fichier d'analyse IA récent (< 24h) si disponible.
Si --analysis est fourni explicitement, il prime sur la détection auto.

Workflow typique :
    1. python scrape_classement.py --nom "ANTOINE LE MORVAN" --no-extra
       → classements_H_40-49_TC_ANTOINE_LE_MORVAN_no_extra.json

    2. (dans une conversation Claude) lire le JSON, écrire analysis_ANTOINE_LE_MORVAN.json

    3. python generate_pdf.py data.json          ← détecte auto l'analyse si < 24h
       python generate_pdf.py data.json --analysis analysis.json  ← force un fichier

Usage :
    python generate_pdf.py data.json
    python generate_pdf.py data.json --analysis analysis.json
    python generate_pdf.py data.json --analysis analysis.json --output rapport.pdf
    python generate_pdf.py data.json --no-analysis  (force sans analyse IA)
"""

import argparse
import json
import os
import sys
import time

from scrape_classement import (
    compute_analysis, _compute_zones,
    generate_pdf, setup_style,
)


# ── Chargement des données scrapées ──────────────────────────────────────────

def load_data(filepath):
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    meta      = data["meta"]
    nom       = meta.get("nom")
    categorie = meta.get("categorie")
    nom_lower = nom.lower() if nom else None

    summary_rows = [
        (s["label"], s["cat_rank"], s["cat_total"], s["resultat"], s["is_nom"])
        for s in data["summary"]
    ]

    event_data = {
        label: [tuple(row) for row in rows]
        for label, rows in data["events"].items()
    }

    return nom, nom_lower, categorie, summary_rows, event_data


# ── Détection automatique du fichier d'analyse ───────────────────────────────

def find_fresh_analysis(json_file, nom):
    """
    Cherche un fichier analysis_NOM.json récent (< 24h) dans le même dossier.
    Essaie plusieurs variantes de casse/format du nom.
    Retourne le chemin si trouvé et frais, None sinon.
    """
    if not nom:
        return None

    base_dir = os.path.dirname(os.path.abspath(json_file))
    # Variantes possibles du nom dans le nom de fichier
    variants = set()
    for n in [nom, nom.upper()]:
        variants.add(n.replace(" ", "_"))

    for name_key in variants:
        path = os.path.join(base_dir, f"analysis_{name_key}.json")
        if os.path.exists(path):
            age_h = (time.time() - os.path.getmtime(path)) / 3600
            if age_h < 24:
                return path
            else:
                print(f"  ⚠  Analyse trouvée mais > 24h ({age_h:.0f}h) : {os.path.basename(path)}")
                print(f"     Demandez une nouvelle analyse pour l'intégrer.")
    return None


# ── Chargement de l'analyse ───────────────────────────────────────────────────

def load_analysis(filepath):
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Génère un PDF C7DC depuis un JSON, avec analyse IA optionnelle."
    )
    parser.add_argument("json_file",
                        help="Fichier JSON produit par scrape_classement.py")
    parser.add_argument("--analysis", metavar="FILE",
                        help="Forcer un fichier JSON d'analyse (prime sur la détection auto)")
    parser.add_argument("--no-analysis", action="store_true",
                        help="Ignorer toute analyse IA (même si un fichier récent existe)")
    parser.add_argument("--output", metavar="FILE",
                        help="Nom du fichier PDF de sortie (défaut : même base que le JSON)")
    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Fichier introuvable : {args.json_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Chargement des données : {args.json_file}")
    nom, nom_lower, categorie, summary_rows, event_data = load_data(args.json_file)
    print(f"  Participant : {nom or '(tous)'}  |  Catégorie : {categorie or 'toutes'}")
    print(f"  {len(event_data)} épreuve(s) : {', '.join(event_data)}")

    # Résolution du fichier d'analyse
    ai_analysis = None
    if args.no_analysis:
        print("  Analyse IA : désactivée (--no-analysis)")
    elif args.analysis:
        if not os.path.exists(args.analysis):
            print(f"Fichier d'analyse introuvable : {args.analysis}", file=sys.stderr)
            sys.exit(1)
        print(f"  Analyse IA : {args.analysis} (explicite)")
        ai_analysis = load_analysis(args.analysis)
    else:
        detected = find_fresh_analysis(args.json_file, nom)
        if detected:
            age_min = (time.time() - os.path.getmtime(detected)) / 60
            print(f"  Analyse IA : {os.path.basename(detected)} (auto-détectée, {age_min:.0f} min)")
            ai_analysis = load_analysis(detected)
        else:
            print("  Analyse IA : aucune (passez un fichier --analysis ou demandez une analyse)")

    output_file = args.output or args.json_file.replace(".json", ".pdf")

    print("\nGénération du PDF...")
    setup_style()
    generate_pdf(output_file, summary_rows, event_data, nom, nom_lower, categorie,
                 ai_analysis=ai_analysis)


if __name__ == "__main__":
    main()
