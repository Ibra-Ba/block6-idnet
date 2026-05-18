"""
ID Check Demo — Détection de fraude documentaire
Bloc 6 IDNet | EfficientNet-B0 + MLflow + Grad-CAM
"""

import io
import logging
import os

import cv2
import mlflow.pytorch
import numpy as np
import streamlit as st
import torch
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from PIL import Image

from src.data.dataset import VAL_TF
from src.models.config import DEVICE
from src.models.efficientnet import FraudClassifier

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "IDNet-Fraud-Detector")
ALIAS = "champion"

# ****   Page config **********
st.set_page_config(
    page_title="ID Check Demo",
    page_icon="🛂",
    layout="centered",
)

# ------ Session state  ------------------------------------------
if "model" not in st.session_state:
    st.session_state.model = None
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.25
if "model_version" not in st.session_state:
    st.session_state.model_version = "unknown"


# ------ Chargement du modèle ----------------

@st.cache_resource(show_spinner=False)
def load_model():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI non défini")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    version = mv.version
    threshold = float(mv.tags.get("deployment_threshold", 0.25))

    model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
    model = mlflow.pytorch.load_model(model_uri, map_location=DEVICE)
    model.eval()

    return model, threshold, version


# ------Inférence ------------------

def predict(model, image_bytes: bytes, threshold: float) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(image)
    tensor = VAL_TF(image=arr)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    fraud_prob = float(probs[1])
    label = "fraud" if fraud_prob >= threshold else "genuine"
    confidence = fraud_prob if label == "fraud" else float(probs[0])

    return {
        "label": label,
        "fraud_probability": round(fraud_prob, 4),
        "confidence": round(confidence, 4),
        "threshold_used": threshold,
    }


# ------------------Grad-CAM -------------------------------------------------

def generate_gradcam(model, image_bytes: bytes) -> np.ndarray:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(image)
    tensor = VAL_TF(image=arr)["image"].unsqueeze(0).to(DEVICE)

    # Image normalisée pour la visualisation
    input_image = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    input_image = input_image - input_image.min()
    input_image = input_image / (input_image.max() + 1e-8)

    target_layers = [model.backbone.blocks[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=tensor)[0]

    return show_cam_on_image(input_image, grayscale_cam, use_rgb=True)


# --------- UI --------------------------------------------

st.title("🛂 ID Check Demo")
st.caption("Détection de fraude documentaire — EfficientNet-B0")
st.divider()

# Chargement du modèle
with st.spinner("Chargement du modèle @champion..."):
    try:
        model, threshold, version = load_model()
        st.success(f"✅ Modèle V{version} chargé (threshold={threshold:.4f})")
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle : {e}")
        st.stop()

# Upload
st.subheader("📄 Document à analyser")
uploaded_file = st.file_uploader(
    "Glissez une image de document d'identité",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))

    st.image(image, caption="Document uploadé", use_column_width=True)
    st.divider()

    # Prédiction
    with st.spinner("Analyse en cours..."):
        result = predict(model, image_bytes, threshold)

    label = result["label"]
    fraud_prob = result["fraud_probability"]

    # -------------- Résultat ---------------
    st.subheader("🔎 Résultat")

    col1, col2, col3 = st.columns(3)
    col1.metric("Score de fraude", f"{fraud_prob:.2%}")
    col2.metric("Seuil", f"{threshold:.2%}")
    col3.metric("Confiance", f"{result['confidence']:.2%}")

    st.progress(int(fraud_prob * 100))

    if label == "fraud":
        st.error("⚠️ Document frauduleux détecté")
    else:
        st.success("✅ Document authentique")

    # ------- Grad-CAM ----------------
    if label == "fraud":
        st.divider()
        st.subheader("🧠 Zones suspectes (Grad-CAM)")
        st.caption("Les zones en rouge indiquent les régions ayant influencé la décision.")

        with st.spinner("Génération de la carte d'activation..."):
            try:
                cam_image = generate_gradcam(model, image_bytes)
                st.image(cam_image, caption="Carte d'activation Grad-CAM", use_column_width=True)
            except Exception as e:
                st.warning(f"Grad-CAM indisponible : {e}")

    # ------------ Détails ---------------
    with st.expander("📊 Détails de la prédiction"):
        st.json({**result, "model_version": version})

st.divider()
st.caption(f"Modèle : {MODEL_NAME} @{ALIAS} | Version : {version} | Device : {DEVICE}")