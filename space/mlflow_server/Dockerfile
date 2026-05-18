# ===================================================================
# Builder stage
# ===================================================================
FROM python:3.11.8-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Installation des dépendances de compilation pour psycopg2 et MLflow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

ENV MLFLOW_HTTP_LINESPACE=true

WORKDIR /build
COPY requirements.txt .

# On installe directement dans un dossier cible pour le copier ensuite
RUN pip install --upgrade pip \
    && pip install --prefix=/install -r requirements.txt

# ===================================================================
# Runtime stage — Hugging Face Spaces compatible
# ===================================================================
FROM python:3.11.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AWS_EC2_METADATA_DISABLED=true \
    PORT=7860

# Installation des librairies runtime pour Postgres (libpq) et Curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Création de l'utilisateur non-root (UID 10001 pour HF)
RUN useradd --create-home --uid 10001 appuser
WORKDIR /home/app

# Récupération des packages installés dans le builder
COPY --from=builder /install /usr/local

# Copie et droits du point d'entrée
COPY --chmod=0755 entrypoint.sh /entrypoint.sh

# Sécurité : on s'assure que l'utilisateur possède son répertoire
RUN chown -R appuser:appuser /home/app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]