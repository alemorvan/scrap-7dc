#!/usr/bin/env python3
"""
Leçon 3 — Apprentissage supervisé.

Question concrète et utile : peut-on prédire le score % d'un athlète sur
une épreuve qu'il N'A PAS FAITE, à partir de ses résultats sur les autres
épreuves ? C'est une question à laquelle le rule-based actuel ne sait pas
répondre du tout (il ne peut rien dire sur une épreuve non tentée).

Contrairement au clustering (leçon 2, non supervisé, pas de "bonne
réponse"), ici on a un vrai LABEL : le score réellement obtenu sur
l'épreuve cible. Le modèle apprend à le retrouver à partir des autres
scores, et on peut vérifier objectivement s'il se trompe ou non.

Usage:
    source ../venv/bin/activate  (depuis ml/)
    python 03_supervised_regression.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scrape_classement import EVENT_ORDER  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"

# L'épreuve qu'on essaie de prédire. Choisie parce qu'elle est bien
# corrélée avec toutes les autres (cf. matrice de corrélation, leçon 2) :
# ni trop courte ni trop longue, un bon "point milieu".
TARGET_EVENT = "2000 m"
FEATURE_EVENTS = [e for e in EVENT_ORDER if e != TARGET_EVENT]


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "dataset_athletes.csv")


def build_xy(df: pd.DataFrame):
    """
    On ne garde que les athlètes qui ONT fait l'épreuve cible (sinon on
    n'a pas de vraie réponse à comparer — on ne peut pas s'entraîner sur
    une valeur inventée). Les features, elles, peuvent contenir des NaN
    (épreuves non faites) : elles seront imputées séparément.
    """
    has_target = df[TARGET_EVENT].notna()
    X = df.loc[has_target, FEATURE_EVENTS]
    y = df.loc[has_target, TARGET_EVENT]
    return X, y


def evaluate(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    """
    Deux métriques :
    - MAE (Mean Absolute Error) : erreur moyenne, dans la même unité que
      y (des points de score %). Facile à interpréter : "en moyenne, on
      se trompe de X points".
    - R² : entre -inf et 1. Proportion de la variation de y que le
      modèle explique. 1 = parfait. 0 = pas mieux que prédire la
      moyenne à chaque fois. Négatif = pire que la moyenne.

    On calcule les deux sur le TRAIN et sur le TEST. Si le modèle est
    excellent en train mais nettement moins bon en test, c'est le
    signal classique d'overfitting : il a appris les exemples par
    cœur plutôt que d'apprendre une règle générale.
    """
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    result = {
        "model": name,
        "MAE_train": mean_absolute_error(y_train, pred_train),
        "MAE_test": mean_absolute_error(y_test, pred_test),
        "R2_train": r2_score(y_train, pred_train),
        "R2_test": r2_score(y_test, pred_test),
    }
    return result, model


def main():
    df = load_dataset()
    X, y = build_xy(df)
    print(f"Épreuve cible : {TARGET_EVENT}")
    print(f"{len(X)} athlètes ont fait cette épreuve (et servent d'exemples).\n")

    # Split 80/20 : on entraîne sur 80% des athlètes, on vérifie sur les
    # 20% restants que le modèle N'A JAMAIS VUS pendant l'entraînement.
    # C'est la seule façon honnête de savoir si un modèle généralise.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train : {len(X_train)} athlètes  |  Test : {len(X_test)} athlètes\n")

    # Imputation des NaN dans les features — IMPORTANT : on calcule les
    # moyennes UNIQUEMENT sur le train, puis on les applique au test.
    # Si on calculait la moyenne sur tout (train+test), des infos du
    # test "fuiteraient" dans l'entraînement (data leakage) et on
    # obtiendrait un score de test trop optimiste, donc trompeur.
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    results = []

    # 0. Baseline : prédire toujours la moyenne, quoi qu'il arrive.
    #    Indispensable pour juger si nos modèles apprennent vraiment
    #    quelque chose, ou s'ils ne font pas mieux qu'une réponse bête.
    r, _ = evaluate("Baseline (moyenne)", DummyRegressor(strategy="mean"),
                     X_train_imp, X_test_imp, y_train, y_test)
    results.append(r)

    # 1. Régression linéaire : le modèle le plus simple. Il cherche les
    #    coefficients d'une somme pondérée des 9 autres scores qui
    #    approche au mieux le score de la cible.
    r, lin_model = evaluate("Régression linéaire", LinearRegression(),
                             X_train_imp, X_test_imp, y_train, y_test)
    results.append(r)

    # 2. Random Forest : modèle beaucoup plus flexible (une "forêt" de
    #    règles de décision). Plus puissant, mais plus susceptible de
    #    surapprendre sur un petit dataset.
    r, rf_model = evaluate(
        "Random Forest", RandomForestRegressor(n_estimators=200, random_state=42),
        X_train_imp, X_test_imp, y_train, y_test,
    )
    results.append(r)

    results_df = pd.DataFrame(results).set_index("model")
    print("=== Comparaison des modèles ===\n")
    with pd.option_context("display.float_format", "{:.2f}".format):
        print(results_df)

    print(
        "\nLecture : si Random Forest a un bien meilleur score en train qu'en "
        "test (contrairement à la régression linéaire), c'est de l'overfitting."
    )

    # Interprétation : quels scores pèsent le plus dans la prédiction ?
    coefs = pd.Series(lin_model.coef_, index=FEATURE_EVENTS).sort_values(key=abs, ascending=False)
    print(f"\nPoids de chaque épreuve dans la prédiction du {TARGET_EVENT} (régression linéaire) :")
    print(coefs.round(2))

    out_path = DATA_DIR / "supervised_results.csv"
    results_df.to_csv(out_path)
    print(f"\nRésultats sauvegardés : {out_path}")


if __name__ == "__main__":
    main()
