#!/usr/bin/env sh
set -e

# Vérification des variables
: "${PORT:?PORT is required}"
: "${BACKEND_STORE_URI:?BACKEND_STORE_URI is required}"
: "${ARTIFACT_ROOT:?ARTIFACT_ROOT is required}"

# On ajoute l'option pour ignorer la vérification stricte du host
exec python -m mlflow server \
  --host 0.0.0.0 \
  --port "$PORT" \
  --backend-store-uri "$BACKEND_STORE_URI" \
  --default-artifact-root "$ARTIFACT_ROOT" \
  --serve-artifacts \
  --gunicorn-opts "--timeout 60 --forwarded-allow-ips='*' --proxy-allow-from='*' --limit-request-line 0"