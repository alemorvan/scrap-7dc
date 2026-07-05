#!/usr/bin/env python3
"""
Application web C7DC — scraping + génération PDF à la demande.
"""

import json
import logging
import os
import threading
import time
import urllib.parse
import uuid
from datetime import datetime
from pathlib import Path

import requests as http_requests
from bs4 import BeautifulSoup
from flask import Flask, abort, jsonify, render_template, request, send_file

from scrape_classement import (
    EXTRA_LABELS,
    auto_detect_category,
    fetch_page,
    generate_pdf,
    get_epreuves,
    get_session,
    is_competition,
    parse_all_tables,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── Configuration (variables d'environnement) ─────────────────────────────────
CACHE_TTL_MINUTES = int(os.environ.get("CACHE_TTL_MINUTES", "30"))
CACHE_TTL  = CACHE_TTL_MINUTES * 60
CACHE_DIR  = Path(os.environ.get("CACHE_DIR", "./cache"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/tmp/c7dc_pdfs"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

log.info(
    "Config — CACHE_TTL=%d min  CACHE_DIR=%s  OUTPUT_DIR=%s",
    CACHE_TTL_MINUTES, CACHE_DIR, OUTPUT_DIR,
)

# ── Couche de cache JSON sur disque ───────────────────────────────────────────

def _cache_read(name: str):
    """
    Lit le cache JSON local.
    Retourne (data, age_secondes) ou (None, +inf) si absent/corrompu.
    """
    path = CACHE_DIR / f"{name}.json"
    try:
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            age = time.time() - stored.get("ts", 0)
            return stored["data"], age
    except Exception as exc:
        log.warning("Lecture cache '%s': %s", name, exc)
    return None, float("inf")


def _cache_write(name: str, data: list):
    """Écrit le cache JSON local."""
    path = CACHE_DIR / f"{name}.json"
    try:
        payload = {
            "ts":    time.time(),
            "date":  datetime.now().isoformat(timespec="seconds"),
            "count": len(data),
            "data":  data,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Cache '%s' sauvegardé : %d entrées → %s", name, len(data), path)
    except Exception as exc:
        log.warning("Écriture cache '%s': %s", name, exc)


# ── Cache participants ────────────────────────────────────────────────────────
_participants_lock = threading.Lock()
_participants: list = []
_participants_ts: float = 0


def _scrape_participants() -> list:
    """Scrape les premières épreuves pour construire l'index de recherche."""
    session = get_session()
    html = fetch_page(session, 1)
    epreuves = get_epreuves(html)

    seen: dict = {}
    scraped = 0
    for eid, label in epreuves.items():
        if is_competition(label) or label in EXTRA_LABELS:
            continue
        try:
            html = fetch_page(session, eid, delay=(scraped > 0))
            for _, nom, categorie, _ in parse_all_tables(html):
                key = nom.lower()
                if key not in seen:
                    seen[key] = {"nom": nom, "categorie": categorie}
            scraped += 1
        except Exception as exc:
            log.warning("Fetch participants %s: %s", label, exc)
        if scraped >= 4:
            break

    log.info("Participants scrapés : %d depuis %d épreuves", len(seen), scraped)
    return list(seen.values())


def _refresh_participants() -> list:
    """Scrape → écrit JSON → met à jour la mémoire. Thread-safe."""
    global _participants, _participants_ts
    data = _scrape_participants()
    _cache_write("participants", data)
    with _participants_lock:
        _participants = data
        _participants_ts = time.time()
    return data


def _get_participants() -> list:
    """Retourne les participants depuis la mémoire, le fichier ou le scraping."""
    global _participants, _participants_ts
    with _participants_lock:
        if _participants and time.time() - _participants_ts < CACHE_TTL:
            return list(_participants)

    # Mémoire expirée → essai fichier JSON
    data, age = _cache_read("participants")
    if data is not None and age < CACHE_TTL:
        log.info("Participants: cache fichier utilisé (%.0f min)", age / 60)
        with _participants_lock:
            _participants = data
            _participants_ts = time.time() - age
        return list(data)

    # Fichier absent ou expiré → scrape
    return _refresh_participants()


# ── Cache clubs ───────────────────────────────────────────────────────────────
CLUBS_URL = "https://c7dc.ffaviron.fr/classement/clubs"
CLUB_URL  = "https://c7dc.ffaviron.fr/classement/club"

_clubs_lock = threading.Lock()
_clubs: list = []
_clubs_ts: float = 0


def _scrape_clubs() -> list:
    """Scrape le classement des clubs."""
    r = http_requests.get(CLUBS_URL, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    clubs = []
    table = soup.find("table")
    if not table:
        return clubs
    for i, row in enumerate(table.find("tbody").find_all("tr"), 1):
        cells = row.find_all("td")
        link  = row.find("a", href=True)
        if len(cells) < 3 or not link:
            continue
        qs = dict(urllib.parse.parse_qsl(link["href"].split("?", 1)[-1]))
        clubs.append({
            "rang":      i,
            "nom":       cells[1].get_text(strip=True),
            "points":    cells[2].get_text(strip=True).replace("\xa0", "").replace(" ", ""),
            "club_id":   qs.get("club_id", ""),
            "season_id": qs.get("season_id", "8"),
        })
    log.info("Clubs scrapés : %d", len(clubs))
    return clubs


def _refresh_clubs() -> list:
    global _clubs, _clubs_ts
    data = _scrape_clubs()
    _cache_write("clubs", data)
    with _clubs_lock:
        _clubs = data
        _clubs_ts = time.time()
    return data


def _get_clubs() -> list:
    global _clubs, _clubs_ts
    with _clubs_lock:
        if _clubs and time.time() - _clubs_ts < CACHE_TTL:
            return list(_clubs)

    data, age = _cache_read("clubs")
    if data is not None and age < CACHE_TTL:
        log.info("Clubs: cache fichier utilisé (%.0f min)", age / 60)
        with _clubs_lock:
            _clubs = data
            _clubs_ts = time.time() - age
        return list(data)

    return _refresh_clubs()


# ── Cache épreuves (toutes les lignes, toutes catégories) ────────────────────
_events_lock = threading.Lock()
_events: dict = {}   # {str(eid): {"label": label, "rows": [[rank, nom, cat, time], ...]}}
_events_ts: float = 0


def _scrape_events() -> dict:
    """Scrape toutes les épreuves et retourne les données brutes (sans filtre catégorie)."""
    _warmup_status["events"] = []   # réinitialise à chaque scrape

    session  = get_session()
    html     = fetch_page(session, 1)
    epreuves = {eid: label for eid, label in get_epreuves(html).items()
                if not is_competition(label) and label not in EXTRA_LABELS}

    result = {}
    for i, (eid, label) in enumerate(epreuves.items()):
        try:
            html = fetch_page(session, eid, delay=(i > 0))
            rows = parse_all_tables(html)   # pas de filtre → toutes catégories
            result[str(eid)] = {"label": label, "rows": [list(r) for r in rows]}
            _warmup_status["events"].append({"label": label, "count": len(rows), "ok": True})
            log.info("Events: %s → %d lignes", label, len(rows))
        except Exception as exc:
            _warmup_status["events"].append({"label": label, "ok": False})
            log.warning("Fetch events %s: %s", label, exc)

    log.info("Events scrapés : %d épreuves", len(result))
    return result


def _refresh_events() -> dict:
    global _events, _events_ts
    data = _scrape_events()
    # Sérialiser comme liste d'objets pour _cache_write
    as_list = [{"eid": eid, **ev} for eid, ev in data.items()]
    _cache_write("events", as_list)
    with _events_lock:
        _events = data
        _events_ts = time.time()
    return data


def _get_events() -> dict:
    global _events, _events_ts
    with _events_lock:
        if _events and time.time() - _events_ts < CACHE_TTL:
            return dict(_events)

    stored, age = _cache_read("events")
    if stored is not None and age < CACHE_TTL:
        log.info("Events: cache fichier utilisé (%.0f min)", age / 60)
        data = {item["eid"]: {"label": item["label"], "rows": item["rows"]} for item in stored}
        with _events_lock:
            _events = data
            _events_ts = time.time() - age
        return data

    return _refresh_events()


# ── Membres d'un club (pas de cache — donnée légère, chargée à la demande) ───

def _load_club_members(club_id: str, season_id: str = "8") -> dict:
    url = f"{CLUB_URL}?club_id={club_id}&season_id={season_id}"
    r = http_requests.get(url, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    club_nom = ""
    for h1 in soup.find_all("h1"):
        txt = h1.get_text(strip=True)
        if "détails du club" in txt.lower():
            club_nom = txt.split(":", 1)[-1].strip() if ":" in txt else txt
            break

    members = []
    table = soup.find("table")
    if table:
        for i, row in enumerate(table.find("tbody").find_all("tr"), 1):
            cells = row.find_all("td")
            if len(cells) < 11:
                continue
            members.append({
                "rang":  i,
                "nom":   cells[1].get_text(strip=True),
                "total": cells[9].get_text(strip=True),
                "defis": cells[10].get_text(strip=True),
            })

    return {"club_nom": club_nom, "members": members}


# ── Warmup initial (une seule fois au démarrage) ─────────────────────────────
# Après warmup, le cache est rafraîchi paresseusement à la première requête
# qui constate que le TTL est dépassé — pas de scraping à vide en arrière-plan.

_app_ready = False
_warmup_status: dict = {"steps": [], "current": None, "events": []}


def _warmup():
    global _app_ready
    log.info("Warmup cache (TTL = %d min)...", CACHE_TTL_MINUTES)

    steps = [
        ("participants", "Participants", _get_participants),
        ("clubs",        "Clubs",        _get_clubs),
        ("events",       "Épreuves",     _get_events),
    ]
    for key, label, fn in steps:
        _warmup_status["current"] = label
        try:
            data = fn()
            count = len(data) if isinstance(data, (list, dict)) else 0
            _warmup_status["steps"].append({"label": label, "count": count, "ok": True})
            log.info("Warmup '%s' OK", key)
        except Exception as exc:
            _warmup_status["steps"].append({"label": label, "ok": False, "error": str(exc)})
            log.error("Warmup '%s' échoué : %s", key, exc)

    _warmup_status["current"] = None
    _app_ready = True
    log.info("Application prête.")


threading.Thread(target=_warmup, daemon=True, name="cache-warmup").start()


# ── Gestion des jobs PDF ──────────────────────────────────────────────────────
_jobs_lock = threading.Lock()
_jobs: dict = {}
_pdf_lock  = threading.Lock()


def _update_job(job_id, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


def _run_generation(job_id, nom, categorie):
    """Génère le PDF à partir du cache des épreuves (scrape si cache absent)."""
    try:
        nom_lower = nom.lower()

        # ── Catégorie ─────────────────────────────────────────────────────────
        if not categorie:
            _update_job(job_id, status="scraping", step="Détection de la catégorie...")
            # 1. cherche dans le cache participants
            participants = _get_participants()
            match = next((p for p in participants if p["nom"].lower() == nom_lower), None)
            if match and match.get("categorie"):
                categorie = match["categorie"]
            else:
                # 2. fallback : scrape live
                session  = get_session()
                html     = fetch_page(session, 1)
                epreuves = get_epreuves(html)
                categorie = auto_detect_category(session, epreuves, nom_lower)
            log.info("Job %s: catégorie = %r", job_id, categorie)

        # ── Données des épreuves depuis le cache ──────────────────────────────
        _update_job(job_id, status="scraping", step="Chargement des résultats...")
        all_events = _get_events()   # {str(eid): {"label": ..., "rows": [...]}}

        summary_rows, event_data = [], {}
        n = len(all_events)

        for i, (eid_str, ev) in enumerate(all_events.items()):
            label    = ev["label"]
            all_rows = ev["rows"]
            _update_job(job_id, step=f"Analyse {label} ({i + 1}/{n})...")

            rows = [r for r in all_rows if r[2] == categorie] if categorie else all_rows
            if not rows:
                continue

            cat_rank, nom_resultat = "-", "-"
            for idx, row in enumerate(rows, 1):
                if row[1].lower() == nom_lower:
                    cat_rank, nom_resultat = idx, row[3]
                    break

            summary_rows.append((label, cat_rank, len(rows), nom_resultat, cat_rank != "-"))
            event_data[label] = rows

        if not event_data:
            raise ValueError(f"Aucune donnée trouvée pour « {nom} »")

        _update_job(job_id, status="generating", step="Génération du PDF...")

        filename    = f"C7DC_{nom.replace(' ', '_').upper()}.pdf"
        output_path = str(OUTPUT_DIR / f"{job_id}_{filename}")

        with _pdf_lock:
            generate_pdf(output_path, summary_rows, event_data, nom, nom_lower, categorie)

        _update_job(job_id, status="done", step="PDF prêt !",
                    file_path=output_path, filename=filename)
        log.info("Job %s terminé : %s", job_id, output_path)

    except Exception as exc:
        log.exception("Job %s échoué", job_id)
        _update_job(job_id, status="error", step=str(exc), error=str(exc))


def _cleanup_loop():
    while True:
        time.sleep(300)
        cutoff = time.time() - 3600
        with _jobs_lock:
            to_remove = [jid for jid, j in _jobs.items() if j.get("created_at", 0) < cutoff]
            for jid in to_remove:
                job = _jobs.pop(jid)
                fp  = job.get("file_path")
                if fp and os.path.exists(fp):
                    try:
                        os.unlink(fp)
                    except OSError:
                        pass


threading.Thread(target=_cleanup_loop, daemon=True, name="pdf-cleanup").start()


# ── Routes ────────────────────────────────────────────────────────────────────

_MAINTENANCE_HTML = """<!doctype html>
<html lang="fr"><head>
<meta charset="utf-8">
<title>Démarrage en cours…</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,sans-serif;background:#f0f4f8;display:flex;
       justify-content:center;align-items:center;min-height:100vh}
  .card{background:#fff;border-radius:16px;padding:2rem 2.5rem;
        box-shadow:0 4px 24px rgba(0,0,0,.1);max-width:460px;width:100%}
  h2{color:#1a73e8;margin-bottom:1.25rem;font-size:1.3rem;text-align:center}
  .spinner{width:36px;height:36px;border:4px solid #e0e7ef;
           border-top-color:#1a73e8;border-radius:50%;
           animation:spin 1s linear infinite;margin:0 auto 1.25rem}
  @keyframes spin{to{transform:rotate(360deg)}}
  .steps{list-style:none;font-size:.9rem}
  .steps li{padding:.35rem 0;border-bottom:1px solid #f0f4f8;display:flex;
            align-items:baseline;gap:.5rem}
  .steps li:last-child{border:none}
  .ok{color:#28a745}.ko{color:#dc3545}.spin{color:#1a73e8}
  .sub{list-style:none;margin:.25rem 0 0 1.2rem;font-size:.8rem;color:#666}
  .sub li{padding:.15rem 0}
  .hint{text-align:center;font-size:.75rem;color:#aaa;margin-top:1rem}
</style>
</head><body>
<div class="card">
  <div class="spinner"></div>
  <h2>Démarrage en cours…</h2>
  <ul class="steps" id="steps"></ul>
  <p class="hint" id="hint">Chargement des données initiales…</p>
</div>
<script>
const stepsEl = document.getElementById('steps');
const hintEl  = document.getElementById('hint');

function icon(ok) { return ok ? '<span class="ok">✓</span>' : '<span class="ko">✗</span>'; }

async function poll() {
  try {
    const r = await fetch('/api/startup_status');
    const d = await r.json();

    function eventsSubList(events, inProgress) {
      if (!events || !events.length) {
        return inProgress ? '<ul class="sub"><li><span class="spin">⏳</span> connexion au site…</li></ul>' : '';
      }
      let s = '<ul class="sub">';
      for (const e of events) {
        s += '<li>' + icon(e.ok) + ' ' + e.label + (e.ok ? ' — ' + e.count + ' lignes' : ' — erreur') + '</li>';
      }
      if (inProgress) s += '<li><span class="spin">⏳</span> en cours…</li>';
      s += '</ul>';
      return s;
    }

    let html = '';
    for (const s of d.steps) {
      const cnt = s.count !== undefined ? ' <small style="color:#999">(' + s.count + ')</small>' : '';
      html += '<li>' + icon(s.ok) + ' ' + s.label + cnt;
      if (s.label === 'Épreuves') html += eventsSubList(d.events, false);
      html += '</li>';
    }
    if (d.current) {
      html += '<li><span class="spin">⏳</span> ' + d.current + '…';
      if (d.current === 'Épreuves') html += eventsSubList(d.events, true);
      html += '</li>';
    }
    stepsEl.innerHTML = html;

    if (d.ready) {
      hintEl.textContent = 'Prêt ! Redirection…';
      window.location.reload();
    } else {
      hintEl.textContent = 'Mise à jour automatique…';
      setTimeout(poll, 1000);
    }
  } catch(e) {
    setTimeout(poll, 2000);
  }
}
poll();
</script>
</body></html>"""


@app.before_request
def check_ready():
    if not _app_ready and request.path not in ("/health", "/api/startup_status"):
        return _MAINTENANCE_HTML, 503


@app.route("/api/startup_status")
def startup_status():
    return jsonify({
        "ready":   _app_ready,
        "current": _warmup_status["current"],
        "steps":   _warmup_status["steps"],
        "events":  _warmup_status["events"],
    })


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    # Indique l'âge des caches dans la réponse
    p_age = round((time.time() - _participants_ts) / 60, 1) if _participants_ts else None
    c_age = round((time.time() - _clubs_ts)        / 60, 1) if _clubs_ts        else None
    return jsonify({
        "status":           "ok" if _app_ready else "starting",
        "ready":            _app_ready,
        "cache_ttl_minutes": CACHE_TTL_MINUTES,
        "participants": {"count": len(_participants), "age_minutes": p_age},
        "clubs":        {"count": len(_clubs),        "age_minutes": c_age},
    })


@app.route("/api/search")
def search():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])
    q_lower = q.lower()
    try:
        participants = _get_participants()
    except Exception as exc:
        log.error("Cache participants: %s", exc)
        return jsonify({"error": str(exc)}), 500
    results = sorted(
        [p for p in participants if q_lower in p["nom"].lower()],
        key=lambda p: p["nom"],
    )
    return jsonify(results[:30])


@app.route("/api/clubs")
def clubs_search():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])
    q_lower = q.lower()
    try:
        all_clubs = _get_clubs()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    results = sorted(
        [c for c in all_clubs if q_lower in c["nom"].lower()],
        key=lambda c: c["rang"],
    )
    return jsonify(results[:20])


@app.route("/api/club_members/<club_id>")
def club_members(club_id):
    season_id = request.args.get("season_id", "8")
    try:
        return jsonify(_load_club_members(club_id, season_id))
    except Exception as exc:
        log.error("club_members %s: %s", club_id, exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True) or {}
    nom  = (data.get("nom") or "").strip()
    if not nom:
        return jsonify({"error": "Nom requis"}), 400

    categorie = (data.get("categorie") or "").strip() or None
    job_id    = str(uuid.uuid4())

    with _jobs_lock:
        _jobs[job_id] = {"status": "pending", "step": "En attente...", "created_at": time.time()}

    threading.Thread(target=_run_generation, args=(job_id, nom, categorie), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def job_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify({
        "status":   job["status"],
        "step":     job.get("step", ""),
        "error":    job.get("error"),
        "filename": job.get("filename"),
    })


@app.route("/api/download/<job_id>")
def download(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job or job.get("status") != "done" or not job.get("file_path"):
        abort(404)
    return send_file(
        job["file_path"],
        as_attachment=True,
        download_name=job["filename"],
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
