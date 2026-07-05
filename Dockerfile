FROM python:3.12-slim

WORKDIR /app

# Dépendances système pour matplotlib (Agg backend)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code de l'application
COPY scrape_classement.py generate_pdf.py app.py ./
COPY templates/ templates/

# Répertoires de travail
RUN mkdir -p /tmp/c7dc_pdfs /app/cache

# Variables configurables
ENV PORT=8080 \
    CACHE_TTL_MINUTES=30 \
    CACHE_DIR=/app/cache \
    OUTPUT_DIR=/tmp/c7dc_pdfs \
    FETCH_DELAY=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["gunicorn", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "1", \
     "--threads", "8", \
     "--timeout", "360", \
     "--log-level", "info", \
     "app:app"]
