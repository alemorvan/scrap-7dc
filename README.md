# C7DC — Analyse de performance aviron

Application web et outil CLI pour scraper les classements du [Challenge des 7 Défis Capitaux](https://c7dc.ffaviron.fr) et générer des rapports PDF personnalisés par participant.

> **Application non officielle**, sans lien avec la Fédération Française d'Aviron ni avec les organisateurs du C7DC.

[![Licence: CC BY-SA 4.0](https://img.shields.io/badge/Licence-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

---

## Fonctionnalités

- Recherche d'un participant par nom ou par club
- Génération de rapport PDF personnalisé :
  - Page de garde et résumé des classements toutes épreuves
  - Courbe d'allure au 500 m (comparaison inter-épreuves)
  - Page détaillée par épreuve (graphique + tableau avec allure /500 m)
  - Tableau des opportunités et préconisations d'entraînement
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

| Variable | Défaut | Description |
|---|---|---|
| `PORT` | `8080` | Port d'écoute |
| `CACHE_TTL_MINUTES` | `30` | Durée de vie du cache en minutes |
| `CACHE_DIR` | `/app/cache` | Répertoire des fichiers JSON de cache |
| `OUTPUT_DIR` | `/tmp/c7dc_pdfs` | Répertoire temporaire des PDF générés |

Exemple avec TTL personnalisé :

```bash
docker run -p 8080:8080 -e CACHE_TTL_MINUTES=60 c7dc-app
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
