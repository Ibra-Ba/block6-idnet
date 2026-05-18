---
title: ID Check Demo
emoji: 🛂
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# 🛂 ID Check Demo

Détection de fraude documentaire par deep learning.

## Modèle
- Architecture : EfficientNet-B0
- Dataset : IDNet ESP (documents d'identité espagnols)
- Classes : `genuine` (authentique) vs `fraud` (frauduleux)
- AUROC : 0.966 | F1 : 0.862

## Utilisation
1. Uploader une image de document d'identité (JPG ou PNG)
2. Le modèle analyse le document et retourne un score de fraude
3. En cas de fraude détectée, une carte Grad-CAM indique les zones suspectes

## Stack
- EfficientNet-B0 (timm)
- MLflow Model Registry
- Streamlit
- Hugging Face Spaces