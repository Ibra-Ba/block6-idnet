"""
Entraînement EfficientNet-B0 — Bloc 6 IDNet
2 phases : freeze backbone → fine-tune partiel
"""

import logging
import os

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import TRAIN_TF, VAL_TF, IDNetDataset
from src.models.config import (
    BATCH_SIZE,
    DEVICE,
    FREEZE_EPOCHS,
    LR_FINETUNE,
    LR_HEAD,
    MIN_AUROC,
    NUM_WORKERS,
    PROCESSED_DIR,
    TOTAL_EPOCHS,
)
from src.models.efficientnet import FraudClassifier

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_model_scores(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            probs = torch.softmax(model(images), dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_probs)


def find_optimal_threshold(y_true, y_score, target_recall=0.95):
    """Seuil maximisant la precision sous contrainte recall >= target_recall."""
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    valid = [
        (p, r, t)
        for p, r, t in zip(precisions, recalls, thresholds)
        if r >= target_recall
    ]
    if not valid:
        logger.warning(f"Recall cible {target_recall} inatteignable → seuil minimal")
        return float(thresholds[0])
    best = max(valid, key=lambda x: x[0])
    logger.info(f"Seuil optimal : {best[2]:.4f} (precision={best[0]:.4f}, recall={best[1]:.4f})")
    return float(best[2])


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    all_labels, all_probs = [], []
    total_loss = 0.0

    desc = "🚀 Train" if is_train else "🧪 Val  "
    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")

    with torch.set_grad_enabled(is_train):
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            total_loss += loss.item() * len(labels)
            pbar.set_postfix({"loss": f"{total_loss/len(all_labels):.3f}"})

    avg_loss = total_loss / len(all_labels)
    auroc = float(roc_auc_score(all_labels, all_probs))
    f1 = float(f1_score(all_labels, (np.array(all_probs) >= 0.5).astype(int)))
    return avg_loss, auroc, f1


# ─── Train ────────────────────────────────────────────────────────────────────

def train():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI non défini")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection"))

    if mlflow.active_run():
        mlflow.end_run()
    os.environ.pop("MLFLOW_RUN_ID", None)

    # ─── DataLoaders ──────────────────────────────────────────────────────────
    train_dl = DataLoader(
        IDNetDataset(PROCESSED_DIR / "train.csv", TRAIN_TF),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_dl = DataLoader(
        IDNetDataset(PROCESSED_DIR / "val.csv", VAL_TF),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # ─── Modèle + loss ────────────────────────────────────────────────────────
    model = FraudClassifier(pretrained=True).to(DEVICE)

    class_counts = np.bincount(
        IDNetDataset(PROCESSED_DIR / "train.csv", TRAIN_TF).df["label"]
    )
    freq = class_counts / class_counts.sum()
    weights = torch.tensor(1.0 / np.sqrt(freq), dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_auroc = 0.0
    patience = 3
    patience_counter = 0
    optimizer = None

    print(f"\n[START] Entraînement sur {DEVICE}")
    print(f"[INFO]  Tracking → {tracking_uri}\n")

    run_id = None
    try:
        with mlflow.start_run(run_name="training") as run:
            run_id = run.info.run_id
            mlflow.log_params({
                "model": "efficientnet_b0",
                "batch_size": BATCH_SIZE,
                "freeze_epochs": FREEZE_EPOCHS,
                "total_epochs": TOTAL_EPOCHS,
                "lr_head": LR_HEAD,
                "lr_finetune": LR_FINETUNE,
            })

            for epoch in range(1, TOTAL_EPOCHS + 1):

                # ─── Gestion des phases ───────────────────────────────────────
                if epoch == 1:
                    model.freeze_backbone()
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LR_HEAD,
                    )
                    print(f"--- Phase 1 : Tête seule (LR={LR_HEAD}) ---")

                elif epoch == FREEZE_EPOCHS + 1:
                    print(f"\n--- Phase 2 : Fine-tuning partiel (LR={LR_FINETUNE}) ---")
                    for name, param in model.backbone.named_parameters():
                        param.requires_grad = any(
                            x in name for x in [
                                "blocks.4", "blocks.5", "blocks.6", "blocks.7", "classifier"
                            ]
                        )
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LR_FINETUNE,
                    )

                # ─── Epoch ───────────────────────────────────────────────────
                tr_loss, tr_auroc, _ = run_epoch(model, train_dl, criterion, optimizer)
                vl_loss, vl_auroc, vl_f1 = run_epoch(model, val_dl, criterion)

                mlflow.log_metrics(
                    {
                        "tr_loss": tr_loss,
                        "vl_loss": vl_loss,
                        "vl_auroc": vl_auroc,
                        "vl_f1": vl_f1,
                    },
                    step=epoch,
                )
                print(
                    f"Epoch {epoch:02d} | "
                    f"tr_loss={tr_loss:.4f} | "
                    f"vl_auroc={vl_auroc:.4f} | "
                    f"vl_f1={vl_f1:.4f}"
                )

                # ─── Checkpoint + early stopping ─────────────────────────────
                if vl_auroc > best_auroc:
                    best_auroc = vl_auroc
                    torch.save(model.state_dict(), "best_model_checkpoint.pt")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"⏳ Patience : {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print("⏹  Early stopping déclenché")
                        break

            # ─── Seuil optimal ────────────────────────────────────────────────
            print("\n--- Calcul du seuil optimal (target recall=95%) ---")
            best_model = FraudClassifier(pretrained=False).to(DEVICE)
            best_model.load_state_dict(
                torch.load("best_model_checkpoint.pt", map_location=DEVICE)
            )
            best_model.eval()

            y_true, y_score = get_model_scores(best_model, val_dl)
            optimal_threshold = find_optimal_threshold(y_true, y_score, target_recall=0.95)

            # ─── Log MLflow ───────────────────────────────────────────────────
            mlflow.pytorch.log_model(
                best_model,
                artifact_path="model",
                pip_requirements="requirements.txt",
            )
            mlflow.log_artifact("best_model_checkpoint.pt", artifact_path="model")
            mlflow.log_metric("best_val_auroc", best_auroc)
            mlflow.log_metric("optimal_threshold", optimal_threshold)
            mlflow.set_tag("optimal_threshold", str(optimal_threshold))

            print(f"\n✅ Modèle loggé — run_id : {run_id}")
            print(f"✅ Seuil optimal : {optimal_threshold:.4f}")

    except KeyboardInterrupt:
        print("\n[STOP] Interruption — best_model_checkpoint.pt conservé")
    finally:
        mlflow.end_run()

    return run_id


if __name__ == "__main__":
    train()