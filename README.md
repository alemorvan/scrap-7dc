# C7DC — Analyse de performance aviron

Application web et outil CLI pour scraper les classements du [Challenge des 7 Défis Capitaux](https://c7dc.ffaviron.fr) et générer des rapports PDF personnalisés par participant.

> **Application non officielle**, sans lien avec la Fédération Française d'Aviron ni avec les organisateurs du C7DC.

[![Licence: CC BY-SA 4.0](https://img.shields.io/badge/Licence-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

---

## Fonctionnalités

- Recherche d'un participant par nom ou par club
- Génération de rapport PDF personnalisé :
  - Page de garde et résumé des classements toutes épreuves
  - Courbe d'allure au 500 m (comparaison inter-épreuves), avec zone de performance attendue (Machine Learning) et zone de concurrence directe
  - Page détaillée par épreuve (graphique + tableau avec allure /500 m), avec zones de concurrence directe (±10 places) et d'extension (11 à 20 places devant)
  - Tableau des opportunités et préconisations d'entraînement
  - Page de prédictions Machine Learning (profil explosif/endurant détecté par clustering, score attendu par épreuve)
- Cache local JSON avec TTL configurable (pas de scraping superflu)
- Page de maintenance automatique pendant le démarrage

---

## Architecture du cache

Au démarrage, l'application charge trois caches dans `CACHE_DIR` :

| Fichier | Contenu | Source |
|---|---|---|
| `participants.json` | Index de recherche (nom + catégorie) | 4 épreuves scrappées |
| `clubs.json` | Classement des 258 clubs | Page clubs C7DC |
| `events.json` | Résultats complets de toutes les épreuves | 10 épreuves scrappées |

Le cache est **lu depuis le disque au redémarrage** si son âge est inférieur au TTL. Il est **rafraîchi paresseusement** à la première requête qui constate son expiration — aucun scraping en arrière-plan à vide.

---

## Installation locale

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Démarrage (développement)

```bash
source venv/bin/activate
gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 8 --timeout 360 app:app
```

Ouvrir [http://localhost:8080](http://localhost:8080).

---

## Docker

### Build

```bash
docker build -t c7dc-app .
```

### Lancement

```bash
docker run -p 8080:8080 c7dc-app
```

### Variables d'environnement

| Variable | Défaut Docker | Défaut local | Description |
|---|---|---|---|
| `PORT` | `8080` | `8080` | Port d'écoute |
| `CACHE_TTL_MINUTES` | `30` | `30` | Durée de vie du cache en minutes |
| `CACHE_DIR` | `/app/cache` | `./cache` | Répertoire des fichiers JSON de cache |
| `OUTPUT_DIR` | `/tmp/c7dc_pdfs` | `/tmp/c7dc_pdfs` | Répertoire temporaire des PDF générés |
| `FETCH_DELAY` | `0` | `1` | Délai en secondes entre chaque requête vers c7dc.ffaviron.fr |

`FETCH_DELAY=0` est recommandé en production (Render, Railway…) pour accélérer le warmup. Par défaut à `1` pour ménager le serveur C7DC lors des tests locaux répétés.

Exemple :

```bash
docker run -p 8080:8080 -e FETCH_DELAY=0 -e CACHE_TTL_MINUTES=60 c7dc-app
```

---

## CI/CD (GitHub Actions)

L'image Docker est automatiquement construite et publiée sur le GitHub Container Registry à chaque push sur `main` :

```
ghcr.io/alemorvan/crap-7dc:latest
```

---

## Déploiement (Render — recommandé)

[Render](https://render.com) est la plateforme recommandée pour l'hébergement :

- **Free tier** : 512 Mo RAM, 0,1 vCPU partagé, 750 h/mois (suffisant pour un service)
- **Comportement** : mise en veille après 15 min d'inactivité, redémarrage en ~30 s
- La page de maintenance gère automatiquement le cold start

### Déploiement

1. Créer un compte sur [render.com](https://render.com)
2. New → Web Service → connecter le dépôt GitHub
3. Sélectionner **Docker** comme environnement
4. Laisser les valeurs par défaut ou ajuster `CACHE_TTL_MINUTES`
5. Deploy

### Alternatives

| Plateforme | Free tier | Avantage |
|---|---|---|
| **Render** | 750 h/mois, sleep après 15 min | Simple, GitHub natif |
| **Fly.io** | 3 VMs 256 Mo, toujours actif | Pas de sleep |
| **Railway** | $5 de crédit/mois | Très simple mais payant |

---

## Machine Learning — profils et prédictions

Le rapport PDF intègre des prédictions statistiques entraînées sur l'ensemble des
résultats scrapés (pas du deep learning à proprement parler : du Machine Learning
classique — clustering + régression linéaire — largement suffisant vu la taille et
la nature des données).

### Ce que ça fait

- **Profil athlétique (clustering K-means)** : détecte automatiquement si un
  athlète est plutôt "Explosif" (fort sur les épreuves courtes) ou "Endurant"
  (fort sur les épreuves longues), sans seuil écrit à la main — contrairement au
  profil rule-based existant (`_detect_profile`).
- **Score attendu par épreuve (régression linéaire)** : pour chaque épreuve,
  prédit le score % qu'un athlète devrait obtenir à partir de ses résultats sur
  les 9 autres épreuves. Permet :
  - d'estimer un score sur une épreuve **jamais tentée** ;
  - de repérer un écart entre score réel et score attendu (signal d'opportunité
    statistique, en complément du tableau des opportunités existant).
- Ces prédictions alimentent la page "Prédictions Machine Learning" du PDF, ainsi
  que les zones affichées sur la courbe d'allure et les pages par épreuve.

**Important** : ce sont des comparaisons statistiques entre athlètes à un instant
donné (un seul scrape) — pas une prédiction de progression dans le temps, et pas
une garantie de résultat.

### Architecture : entraînement (dev) vs inférence (prod)

Le projet sépare volontairement les deux, pour ne pas alourdir l'image Docker de
production :

| | Entraînement | Inférence |
|---|---|---|
| Dossier | `ml/` | racine du projet (`predict_ml.py`) |
| Dépendances | `pandas`, `scikit-learn` (`ml/requirements-ml.txt`) | aucune (stdlib `json`/`math` uniquement) |
| Où ça tourne | Jamais en production, uniquement en local/dev | Dans l'appli Flask/CLI, à chaque génération de PDF |
| Entrée | Un snapshot JSON scrapé (`ml/data/*.json`) | `models/c7dc_model_params.json` |
| Sortie | `models/c7dc_model_params.json` (~8 Ko) | Prédictions (score attendu, profil) |

`scikit-learn`/`pandas` n'apparaissent pas dans `requirements.txt` ni dans le
`Dockerfile` : seul `models/c7dc_model_params.json` (quelques Ko de coefficients
et centres de clusters) est copié dans l'image, et `predict_ml.py` réapplique les
formules à la main (produit scalaire, distance au centre le plus proche).

### Mettre à jour le modèle

Le modèle **ne se réentraîne jamais tout seul**. Il faut le régénérer manuellement
à chaque fois qu'on veut qu'il apprenne de nouvelles données (plus de
participants, une nouvelle saison…) :

```bash
# 1. Installer les dépendances ML (une fois, dans le venv de dev)
source venv/bin/activate
pip install -r ml/requirements-ml.txt

# 2. Récupérer un snapshot frais de toutes les épreuves
python scrape_classement.py          # génère classements.json à la racine
cp classements.json ml/data/classements_$(date +%F).json

# 3. Mettre à jour DATA_FILE dans ml/01_prepare_data.py et ml/04_export_for_production.py
#    pour pointer vers ce nouveau fichier

# 4. Reconstruire le dataset d'entraînement
python ml/01_prepare_data.py

# 5. (Optionnel) Explorer clustering et régression, vérifier que les métriques
#    (silhouette, MAE) restent raisonnables
python ml/02_clustering.py
python ml/03_supervised_regression.py

# 6. Exporter les modèles finaux pour la production
python ml/04_export_for_production.py   # régénère models/c7dc_model_params.json
```

Puis committer `models/c7dc_model_params.json` et redéployer (rebuild de l'image
Docker) — c'est ce fichier, et lui seul, que la production utilise.

### Fichiers concernés

| Fichier | Rôle |
|---|---|
| `ml/01_prepare_data.py` | JSON scrapé → tableau de features (score % par athlète × épreuve) |
| `ml/02_clustering.py` | Exploration + clustering K-means (profils) |
| `ml/03_supervised_regression.py` | Exploration + régression (score attendu) |
| `ml/04_export_for_production.py` | Entraînement final + export JSON léger |
| `models/c7dc_model_params.json` | Paramètres des modèles, utilisés en production |
| `predict_ml.py` | Inférence en production (aucune dépendance ML) |

---

## CLI (mode ligne de commande)

Le script `scrape_classement.py` reste utilisable directement :

```bash
# Générer un PDF pour un participant
python scrape_classement.py --nom "ANTOINE LE MORVAN" --categorie "H 40-49 TC"

# Toutes les épreuves, tous les participants (export Excel)
python scrape_classement.py

# Filtrer par catégorie, sans épreuves longues
python scrape_classement.py --categorie "H 40-49 TC" --no-extra
```

---

## Licence

[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — Antoine Le Morvan
