"""
Évaluation du modèle sur le test set — Bloc 6 IDNet
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.data.dataset import VAL_TF, IDNetDataset
from src.models.config import (
    BATCH_SIZE,
    DEVICE,
    NUM_WORKERS,
    PROCESSED_DIR,
)

load_dotenv()


# ─── Seuil optimal (Youden) ───────────────────────────────────────────────────

def find_optimal_threshold(y_true, y_score):
    """Seuil maximisant l'indice de Youden (TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    optimal_idx = np.argmax(tpr - fpr)
    return float(thresholds[optimal_idx])


# ─── Métriques ────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    return {
        "auroc":          float(roc_auc_score(y_true, y_score)),
        "accuracy":       float(accuracy_score(y_true, y_pred)),
        "f1":             float(f1_score(y_true, y_pred)),
        "threshold_used": threshold,
    }


# ─── Évaluation principale ────────────────────────────────────────────────────

def evaluate(model, loader, artifact_dir: Path | None = None, log_to_mlflow: bool = False):
    model.eval()
    all_labels, all_probs = [], []

    import torch
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            probs = torch.softmax(model(images), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true  = np.array(all_labels)
    y_score = np.array(all_probs)

    threshold = find_optimal_threshold(y_true, y_score)
    print(f"\n[INFO] Seuil optimal (Youden) : {threshold:.4f}")

    metrics = compute_metrics(y_true, y_score, threshold)
    y_pred  = (y_score >= threshold).astype(int)

    print(f"       AUROC    : {metrics['auroc']:.4f}")
    print(f"       Accuracy : {metrics['accuracy']:.4f}")
    print(f"       F1       : {metrics['f1']:.4f}")

    # ─── Artefacts visuels ────────────────────────────────────────────────────
    if artifact_dir or log_to_mlflow:
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=["genuine", "fraud"]).plot(ax=ax_cm)
        ax_cm.set_title(f"Confusion Matrix (threshold={threshold:.2f})")

        # Courbe ROC
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, y_score, ax=ax_roc)
        ax_roc.plot([0, 1], [0, 1], "k--")
        ax_roc.set_title("ROC Curve")

        if log_to_mlflow:
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
            mlflow.log_param("optimal_threshold", threshold)
            mlflow.log_figure(fig_cm, "confusion_matrix.png")
            mlflow.log_figure(fig_roc, "roc_curve.png")

        if artifact_dir:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            fig_cm.savefig(artifact_dir / "confusion_matrix.png")
            fig_roc.savefig(artifact_dir / "roc_curve.png")
            print(f"[OK] Artefacts sauvegardés → {artifact_dir}")

        plt.close(fig_cm)
        plt.close(fig_roc)

    return metrics


# ─── Évaluation depuis un run MLflow ─────────────────────────────────────────

def evaluate_from_run(run_id: str) -> dict:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)

    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model").to(DEVICE)

    test_dl = DataLoader(
        IDNetDataset(PROCESSED_DIR / "test.csv", VAL_TF),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    with mlflow.start_run(run_id=run_id, nested=True):
        return evaluate(model, test_dl, log_to_mlflow=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else os.getenv("MLFLOW_RUN_ID")
    if not run_id:
        print("Usage: python -m src.models.evaluate <run_id>")
        sys.exit(1)

    results = evaluate_from_run(run_id)
    print("\n--- Résultats ---")
    for k, v in results.items():
        print(f"  {k}: {v}")