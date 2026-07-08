#!/usr/bin/env python3
"""
Leçon 1 — Préparer les données.

Un modèle de ML ne mange jamais du JSON brut : il lui faut un TABLEAU
(lignes = exemples, colonnes = features numériques). C'est ce qu'on
appelle la "feature matrix". Ce script construit cette matrice à partir
du JSON scrapé, et l'explique au fur et à mesure.

Usage:
    source ../venv/bin/activate  (depuis ml/)
    python 01_prepare_data.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scrape_classement import classify_event, parse_result, EVENT_ORDER  # noqa: E402

DATA_FILE = Path(__file__).parent / "data" / "classements_2026-07-08.json"
CORE_EVENTS = set(EVENT_ORDER)  # les 10 épreuves "individuelles" (pas les relais/saisons)


def load_raw():
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)["events"]


def build_long_dataframe(events: dict) -> pd.DataFrame:
    """
    Une ligne = un(e) participant(e) sur une épreuve, avec son score
    normalisé dans SA catégorie (même logique que compute_analysis()
    dans scrape_classement.py : (total - rang) / total * 100).

    Pourquoi normaliser en % plutôt que garder le temps brut ?
    Un temps de "1:19.7" au 500m n'est pas comparable à un temps de
    "35:12.4" au 10000m. Le % relatif à la catégorie, lui, est
    comparable event <-> event : c'est notre unité commune.
    """
    records = []
    for label, rows in events.items():
        if label not in CORE_EVENTS:
            continue  # on ignore les relais/24h/saisons pour ce premier jet

        # Regrouper par catégorie EN PRÉSERVANT L'ORDRE (le site trie déjà
        # par performance). La position dans le sous-groupe = le rang
        # catégorie, exactement comme le fait compute_analysis().
        by_cat: dict[str, list] = {}
        for rang_str, nom, categorie, resultat in rows:
            by_cat.setdefault(categorie, []).append((nom, resultat))

        for categorie, entries in by_cat.items():
            total = len(entries)
            if total < 5:
                continue  # catégorie trop petite pour qu'un %  ait du sens
            for rang, (nom, resultat) in enumerate(entries, 1):
                val = parse_result(resultat)
                if val is None:
                    continue
                records.append({
                    "nom": nom,
                    "categorie": categorie,
                    "event": label,
                    "rang": rang,
                    "total": total,
                    "score_pct": (total - rang) / total * 100,
                    "resultat_brut": resultat,
                })

    return pd.DataFrame.from_records(records)


def pivot_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Passe du format long (1 ligne = 1 perf) au format large
    (1 ligne = 1 athlète, 1 colonne = 1 épreuve). C'est le format
    attendu par la quasi-totalité des algos scikit-learn.
    """
    wide = df_long.pivot_table(
        index=["nom", "categorie"],
        columns="event",
        values="score_pct",
        aggfunc="first",
    )
    wide = wide.reindex(columns=EVENT_ORDER)  # ordre stable et lisible
    wide["n_epreuves"] = wide.notna().sum(axis=1)
    return wide.reset_index()


def main():
    print(f"Chargement de {DATA_FILE.name}...")
    events = load_raw()

    df_long = build_long_dataframe(events)
    print(f"\n{len(df_long)} performances individuelles chargées "
          f"sur {df_long['event'].nunique()} épreuves.")

    df_wide = pivot_wide(df_long)
    print(f"{len(df_wide)} athlètes uniques (nom+catégorie) au total.\n")

    print("Répartition du nombre d'épreuves faites par athlète :")
    print(df_wide["n_epreuves"].value_counts().sort_index())

    # Le dataset "propre" qu'on utilisera pour le ML : athlètes avec
    # au moins 3 épreuves, sinon il n'y a pas assez de signal pour
    # dessiner un "profil".
    df_clean = df_wide[df_wide["n_epreuves"] >= 3].copy()
    print(f"\n→ {len(df_clean)} athlètes retenus pour la suite "
          f"(>= 3 épreuves faites).")

    print("\nAperçu (scores %, NaN = épreuve non faite) :")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df_clean.head(8))

    out_path = Path(__file__).parent / "data" / "dataset_athletes.csv"
    df_clean.to_csv(out_path, index=False)
    print(f"\nDataset sauvegardé : {out_path}")


if __name__ == "__main__":
    main()
