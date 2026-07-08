#!/usr/bin/env python3
"""
Inférence des modèles ML entraînés hors-ligne (voir ml/04_export_for_production.py).

Volontairement sans scikit-learn ni pandas : ce module ne fait que réappliquer
à la main les formules d'un modèle déjà entraîné (produit scalaire pour la
régression, distance au centre le plus proche pour le clustering). Ça évite
d'ajouter ~270 Mo de dépendances à l'image Docker de production pour un calcul
qui, une fois le modèle entraîné, est trivial.

Le fichier de paramètres (models/c7dc_model_params.json) est généré par le
pipeline d'entraînement dans ml/ et doit être régénéré (puis redéployé)
lorsque de nouvelles données sont scrapées — il n'y a pas de ré-entraînement
automatique en production.
"""

import json
import math
from pathlib import Path

PARAMS_PATH = Path(__file__).parent / "models" / "c7dc_model_params.json"

_params = None


def _load():
    global _params
    if _params is None:
        with open(PARAMS_PATH, encoding="utf-8") as f:
            _params = json.load(f)
    return _params


def is_available() -> bool:
    return PARAMS_PATH.exists()


def model_info() -> dict:
    p = _load()
    return {"n_athletes": p["source_n_athletes"], "generated_at": p["generated_at"]}


def predict_profile(scores: dict) -> dict | None:
    """
    scores : {label_épreuve: score_pct} pour les épreuves déjà faites par
    l'athlète (les autres seront complétées par la moyenne globale).

    Retourne {"label": "Explosif"|"Endurant", "distances": {...}} ou None
    si les paramètres du modèle sont absents.
    """
    p = _load()
    events = p["event_order"]
    means = p["event_means"]
    cl = p["clustering"]

    x = [scores.get(e, means[e]) for e in events]
    row_mean = sum(x) / len(x)
    x_centered = [v - row_mean for v in x]
    x_scaled = [
        (v - m) / s for v, m, s in zip(x_centered, cl["scaler_mean"], cl["scaler_scale"])
    ]

    best_idx, best_dist = None, math.inf
    distances = {}
    for i, center in enumerate(cl["centers"]):
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(x_scaled, center)))
        distances[cl["cluster_labels"][str(i)]] = round(dist, 2)
        if dist < best_dist:
            best_idx, best_dist = i, dist

    return {"label": cl["cluster_labels"][str(best_idx)], "distances": distances}


def predict_expected_score(scores: dict, target_event: str) -> dict | None:
    """
    Prédit le score % attendu sur `target_event`, à partir des scores
    connus sur les autres épreuves (complétées par la moyenne globale si
    manquantes). Retourne {"predicted": float, "mae": float, "n_athletes": int}
    ou None si aucun modèle n'existe pour cette épreuve (pas assez de données
    à l'entraînement).
    """
    p = _load()
    reg = p["regression"].get(target_event)
    if reg is None:
        return None

    means = p["event_means"]
    x = [scores.get(e, means[e]) for e in reg["features"]]
    pred = reg["intercept"] + sum(c * v for c, v in zip(reg["coef"], x))
    pred = max(0.0, min(100.0, pred))  # un score % reste entre 0 et 100

    return {
        "predicted": round(pred, 1),
        "mae": reg["mae_test"],
        "n_athletes": reg["n_athletes"],
    }
