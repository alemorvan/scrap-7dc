#!/usr/bin/env python3
"""
Leçon 4 — Exporter les modèles pour la production, SANS scikit-learn/pandas
comme dépendance de prod.

Idée : scikit-learn/pandas ne servent qu'à ENTRAÎNER (ici, dans ml/, hors
ligne, jamais déployé). Un modèle entraîné, une fois qu'on ne veut plus que
PRÉDIRE, se résume à quelques nombres :
  - régression linéaire = des coefficients + une constante (un produit
    scalaire à calculer à la main)
  - KMeans = des positions de centres (juste une distance à calculer)

On exporte donc uniquement ces nombres dans un JSON, que l'appli en prod
lira avec le module stdlib `json` — zéro dépendance ML ajoutée à l'image
Docker de production (cf. predict_ml.py à la racine du projet).

Usage:
    source ../venv/bin/activate  (depuis ml/)
    python 04_export_for_production.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scrape_classement import EVENT_ORDER  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
MODELS_OUT = Path(__file__).parent.parent / "models" / "c7dc_model_params.json"

SHORT_EVENTS = {"100 m", "1 min", "500 m"}
LONG_EVENTS = {"30 min", "10000 m", "60 min"}


def export_clustering(df: pd.DataFrame, event_means: dict) -> dict:
    """Ré-entraîne le clustering de la leçon 2 (row-centered, k=2) et
    n'exporte que ce qu'il faut pour classer un NOUVEL athlète : les
    centres, et les paramètres du scaler."""
    X = df[EVENT_ORDER].fillna(pd.Series(event_means))
    X_centered = X.sub(X.mean(axis=1), axis=0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_centered)

    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    km.fit(X_scaled)

    centers = scaler.inverse_transform(km.cluster_centers_)  # retour en "points d'écart"
    labels = {}
    for i, center in enumerate(centers):
        c = dict(zip(EVENT_ORDER, center))
        short_avg = np.mean([c[e] for e in SHORT_EVENTS])
        long_avg = np.mean([c[e] for e in LONG_EVENTS])
        labels[str(i)] = "Explosif" if short_avg > long_avg else "Endurant"

    return {
        "centers": km.cluster_centers_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "cluster_labels": labels,
    }


def export_regressions(df: pd.DataFrame, event_means: dict) -> dict:
    """
    Une régression linéaire par épreuve cible (prédire cette épreuve à
    partir des 9 autres). Pour chacune :
      - on mesure d'abord un MAE de test honnête (split 80/20), pour
        pouvoir afficher une marge d'erreur réaliste dans le PDF
      - puis on ré-entraîne sur 100% des données dispo pour obtenir
        les coefficients finaux réellement utilisés en prod
        (une fois la méthode validée, il n'y a plus de raison de se
        priver de données pour le modèle final)
    """
    regressions = {}
    for target in EVENT_ORDER:
        features = [e for e in EVENT_ORDER if e != target]
        has_target = df[target].notna()
        X = df.loc[has_target, features]
        y = df.loc[has_target, target]

        if len(X) < 30:
            print(f"  [{target}] ignoré : seulement {len(X)} athlètes, pas assez.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        imp = SimpleImputer(strategy="mean").fit(X_train)
        lin_test = LinearRegression().fit(imp.transform(X_train), y_train)
        mae_test = mean_absolute_error(y_test, lin_test.predict(imp.transform(X_test)))

        # Modèle final : ré-entraîné sur 100% des athlètes disponibles.
        imp_full = SimpleImputer(strategy="mean").fit(X)
        lin_full = LinearRegression().fit(imp_full.transform(X), y)

        regressions[target] = {
            "features": features,
            "coef": lin_full.coef_.tolist(),
            "intercept": float(lin_full.intercept_),
            "mae_test": round(float(mae_test), 1),
            "n_athletes": int(len(X)),
        }
        print(f"  [{target}] n={len(X):>4}  MAE_test={mae_test:5.1f}")

    return regressions


def main():
    df = pd.read_csv(DATA_DIR / "dataset_athletes.csv")
    event_means = df[EVENT_ORDER].mean().to_dict()

    print("Entraînement du clustering (profil explosif/endurant)...")
    clustering = export_clustering(df, event_means)
    print(f"  Labels de clusters : {clustering['cluster_labels']}")

    print("\nEntraînement des régressions (une par épreuve cible)...")
    regressions = export_regressions(df, event_means)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_n_athletes": int(len(df)),
        "event_order": EVENT_ORDER,
        "event_means": {k: round(float(v), 2) for k, v in event_means.items()},
        "clustering": clustering,
        "regression": regressions,
    }

    MODELS_OUT.parent.mkdir(exist_ok=True)
    with open(MODELS_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    size_kb = MODELS_OUT.stat().st_size / 1024
    print(f"\nExporté : {MODELS_OUT}  ({size_kb:.1f} Ko)")


if __name__ == "__main__":
    main()
