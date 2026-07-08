#!/usr/bin/env python3
"""
Leçon 2 — Exploration + clustering non supervisé.

Objectif : laisser un algorithme découvrir tout seul des "profils
athlétiques" dans les données, sans lui donner aucune règle — à
comparer avec _detect_profile() (scrape_classement.py) qui, lui,
utilise des seuils écrits à la main (peak > seuil + 10, etc.).

Usage:
    source ../venv/bin/activate  (depuis ml/)
    python 02_clustering.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scrape_classement import EVENT_ORDER  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# Palette catégorielle validée (ordre fixe, cf. skill dataviz) — on prend
# les N premières teintes dans cet ordre, jamais mélangées.
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "dataset_athletes.csv")


# ── 1. Exploration : les épreuves sont-elles corrélées entre elles ? ────────

def explore_correlations(df: pd.DataFrame):
    """
    Un coefficient de corrélation (entre -1 et 1) mesure si deux colonnes
    varient ensemble. Ici : si quelqu'un est bon au 500m, a-t-il tendance
    à être bon aussi au 1000m ? Au 60min ?
    Si oui (corrélation forte) → les deux épreuves mesurent presque la
    même qualité physique. Si non → ce sont deux qualités différentes.
    C'est la vérification statistique de ce que le code fait déjà
    "à la main" avec EFFORT_MULTIPLIER / ENERGY_ZONES.
    """
    corr = df[EVENT_ORDER].corr()
    print("Corrélation entre épreuves (1.0 = parfaitement liées) :\n")
    with pd.option_context("display.width", 160, "display.float_format", "{:.2f}".format):
        print(corr)
    print(
        "\nÀ regarder : les épreuves courtes (100m, 1 min, 500m) sont-elles "
        "plus corrélées entre elles qu'avec les longues (30min, 10000m, 60min) ?"
    )
    return corr


# ── 2. Préparation pour le clustering ────────────────────────────────────────

def prepare_features(df: pd.DataFrame, center_rows: bool = False):
    """
    KMeans ne sait pas gérer les valeurs manquantes (NaN) ni les échelles
    différentes. Deux étapes obligatoires :

    1. Imputation : remplacer chaque NaN par la moyenne de sa colonne.
       C'est une approximation raisonnable ("s'il n'a pas fait le 5000m,
       on suppose qu'il est dans la moyenne des gens qui l'ont fait")
       mais ça reste une approximation — à garder en tête.
    2. Scaling (StandardScaler) : recentrer chaque colonne pour qu'elle
       ait une moyenne de 0 et un écart-type de 1. Nos scores sont déjà
       tous en %, mais leur dispersion (écart-type) diffère d'une épreuve
       à l'autre — sans ce recentrage, l'épreuve la plus "étalée"
       dominerait artificiellement le calcul de distance de KMeans.

    center_rows=True ajoute une 3e étape, AVANT le scaling : on soustrait
    à chaque ATHLÈTE (chaque ligne) sa propre moyenne sur ses épreuves.
    Sans ça, la plus grosse source de variation dans les données est le
    niveau général (bon partout vs moins bon partout), qui écrase le
    signal qu'on veut vraiment capter : la FORME du profil (relativement
    meilleur en sprint ou en endurance, chez CET athlète-là).
    """
    X_raw = df[EVENT_ORDER]

    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=EVENT_ORDER)

    if center_rows:
        X_imputed = X_imputed.sub(X_imputed.mean(axis=1), axis=0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, imputer, scaler


# ── 3. Combien de groupes (k) ? ──────────────────────────────────────────────

def choose_k(X_scaled) -> int:
    """
    KMeans exige de dire à l'avance combien de groupes on veut (k).
    On teste plusieurs valeurs et on regarde deux indicateurs :

    - inertia : à quel point les points sont proches du centre de leur
      groupe (plus bas = mieux, mais baisse toujours plus on augmente k
      → on cherche le "coude" où ça arrête de baisser vite : méthode du coude).
    - silhouette : entre -1 et 1, mesure si les groupes sont bien séparés
      les uns des autres (plus haut = mieux). Contrairement à l'inertia,
      elle ne baisse pas mécaniquement avec k, donc plus fiable pour choisir.
    """
    print("\nRecherche du nombre de profils (k) :")
    print(f"{'k':>3} {'inertia':>12} {'silhouette':>12}")
    best_k, best_score = None, -1
    for k in range(2, 8):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"{k:>3} {km.inertia_:>12.1f} {score:>12.3f}")
        if score > best_score:
            best_k, best_score = k, score
    print(f"\n→ Meilleur score de silhouette pour k = {best_k}")
    return best_k


# ── 4. Entraînement final + interprétation ───────────────────────────────────

def fit_and_describe(df: pd.DataFrame, X_scaled, imputer, scaler, k: int, centered: bool):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    df = df.copy()
    df["cluster"] = km.fit_predict(X_scaled)

    # Les centres de clusters sont en échelle "standardisée" (moyenne 0,
    # écart-type 1) — illisible tel quel. On les "dé-standardise" pour
    # retrouver une unité lisible et pouvoir les interpréter.
    # (si centered=True : ce sont des écarts à la moyenne perso de
    #  l'athlète, pas des % absolus — un 0 = "dans sa moyenne")
    centers_pct = scaler.inverse_transform(km.cluster_centers_)
    centers_df = pd.DataFrame(centers_pct, columns=EVENT_ORDER)
    centers_df["n_athletes"] = df["cluster"].value_counts().sort_index().values

    unit = "écart à sa propre moyenne, en points" if centered else "score % moyen dans la catégorie"
    print(f"\n=== {k} profils découverts automatiquement ({unit}) ===\n")
    with pd.option_context("display.width", 160, "display.float_format", "{:.0f}".format):
        print(centers_df)

    return df, centers_df, km


def plot_profiles(centers_df: pd.DataFrame, out_path: Path, centered: bool, title: str):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(EVENT_ORDER))

    for i, row in centers_df.iterrows():
        color = PALETTE[i % len(PALETTE)]
        y = row[EVENT_ORDER].values
        ax.plot(x, y, marker="o", linewidth=2, markersize=6, color=color)
        # Label direct en bout de ligne plutôt que de dépendre de la légende
        ax.annotate(
            f"Profil {i}  (n={int(row['n_athletes'])})",
            (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
            va="center", fontsize=9, fontweight="bold", color=color,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(EVENT_ORDER, rotation=30, ha="right")
    if centered:
        ax.axhline(0, color="#888", linewidth=1, linestyle="--")
        ax.set_ylabel("Écart à sa propre moyenne (points)")
    else:
        ax.set_ylabel("Score % moyen dans la catégorie")
        ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nGraphique sauvegardé : {out_path}")


def main():
    df = load_dataset()
    explore_correlations(df)

    print("\n" + "=" * 78)
    print("TENTATIVE 1 — clustering sur les scores bruts (niveau général)")
    print("=" * 78)
    X_scaled, imputer, scaler = prepare_features(df, center_rows=False)
    k = choose_k(X_scaled)
    _, centers_df, _ = fit_and_describe(df, X_scaled, imputer, scaler, k, centered=False)
    plot_profiles(
        centers_df, OUT_DIR / "clusters_niveau_general.png", centered=False,
        title="Tentative 1 : clustering brut → capte le NIVEAU, pas la forme",
    )

    print("\n" + "=" * 78)
    print("TENTATIVE 2 — clustering sur les profils centrés (forme du profil)")
    print("=" * 78)
    X_scaled2, imputer2, scaler2 = prepare_features(df, center_rows=True)
    k2 = choose_k(X_scaled2)
    df_clustered, centers_df2, km2 = fit_and_describe(
        df, X_scaled2, imputer2, scaler2, k2, centered=True
    )
    plot_profiles(
        centers_df2, OUT_DIR / "clusters_forme_profil.png", centered=True,
        title="Tentative 2 : profils centrés → capte la FORME (points forts/faibles)",
    )

    df_clustered.to_csv(DATA_DIR / "dataset_athletes_clustered.csv", index=False)
    print(f"\nDataset avec clusters (tentative 2) sauvegardé : "
          f"{DATA_DIR / 'dataset_athletes_clustered.csv'}")


if __name__ == "__main__":
    main()
