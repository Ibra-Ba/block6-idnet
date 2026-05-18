FROM python:3.10-slim

WORKDIR /app

# ─── Dépendances système ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ─── PyTorch CPU ──────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ─── Dépendances Python ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Code source ──────────────────────────────────────────────────────────────
COPY src/ ./src/
COPY app.py .
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# ─── Utilisateur non-root ─────────────────────────────────────────────────────
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# ─── Streamlit ────────────────────────────────────────────────────────────────
EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
    "--server.port", "7860", \
    "--server.address", "0.0.0.0", \
    "--server.enableCORS", "false", \
    "--server.enableXsrfProtection", "false"]