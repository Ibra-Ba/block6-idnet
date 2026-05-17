"""
Enregistrement et promotion du modèle dans MLflow Registry — Bloc 6 IDNet
"""

import os

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()


def register_and_promote(run_id: str, model_name: str = "IDNet-Fraud-Detector"):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # ─── Récupération du threshold depuis le run ──────────────────────────────
    run = client.get_run(run_id)
    threshold = run.data.metrics.get("optimal_threshold")

    if threshold is None:
        threshold_str = run.data.tags.get("optimal_threshold")
        threshold = float(threshold_str) if threshold_str else 0.5
        print(f"[WARN] threshold non trouvé dans metrics → fallback : {threshold}")
    else:
        print(f"[INFO] threshold récupéré depuis metrics : {threshold:.4f}")

    # ─── 1. Enregistrement ────────────────────────────────────────────────────
    print(f"\n[1/3] Enregistrement du run {run_id}...")
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    version = mv.version
    print(f"      Version créée : V{version}")

    # ─── 2. Tags ──────────────────────────────────────────────────────────────
    print(f"[2/3] Ajout des tags à V{version}...")
    client.set_model_version_tag(model_name, version, "optimal_threshold", str(threshold))
    client.set_model_version_tag(model_name, version, "deployment_threshold", str(threshold))
    client.set_model_version_tag(model_name, version, "origin", "bloc6_training")

    # ─── 3. Promotion @champion ───────────────────────────────────────────────
    print(f"[3/3] Promotion V{version} → @champion...")
    client.set_registered_model_alias(model_name, "champion", version)
    print(f"\n✅ V{version} est désormais @champion")
    print(f"   model_name : {model_name}")
    print(f"   threshold  : {threshold:.4f}")

    return version


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.models.register_model <run_id>")
        sys.exit(1)

    run_id = sys.argv[1]
    model_name = os.getenv("MLFLOW_MODEL_NAME", "IDNet-Fraud-Detector")
    register_and_promote(run_id, model_name)
`