# 🛂 Bloc 6 — Détection de Fraude Documentaire

Projet de certification Bac+4 — Direction de projets de gestion de données  
**Jedha Bootcamp | Fullstack**

## Contexte

La fraude documentaire représente un enjeu majeur pour les organisations 
soumises aux obligations KYC (Know Your Customer) et aux réglementations 
eIDAS. Ce projet implémente un système de détection automatique de documents 
d'identité frauduleux par deep learning.

## Démo

👉 [ID Check Demo — Hugging Face Space](https://huggingface.co/spaces/VoxUp/id-check-demo)

## Dataset

- **Source** : [IDNet — Zenodo](https://zenodo.org/records/10462204)
- **Sous-ensemble** : Documents espagnols (ESP)
- **Classes** : `genuine` (authentique) vs `fraud` (frauduleux)
- **6 types de fraude** : copy & move, face morphing, face replacement, 
  combined, inpaint & rewrite, crop & replace
- **Rééchantillonnage** : 30% fraude / 70% genuine

## Modèle

- **Architecture** : EfficientNet-B0 (transfer learning)
- **Entraînement** : 2 phases — freeze backbone → fine-tuning partiel
- **Optimisation du seuil** : contrainte recall ≥ 95% (Youden + métier)
- **Tracking** : MLflow (NeonDB + AWS S3)

## Résultats

| Métrique | Valeur |
|----------|--------|
| AUROC    | 0.966  |
| F1       | 0.862  |
| Seuil déployé | 0.2449 |

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Modèle | EfficientNet-B0 (timm) |
| Tracking | MLflow 2.9 |
| Backend MLflow | NeonDB (PostgreSQL) |
| Artifact store | AWS S3 |
| Déploiement | Hugging Face Spaces |
| UI | Streamlit |
| Explainabilité | Grad-CAM |

## Structure du projet

bloc6-idnet/
├── src/
│   ├── data/
│   │   ├── dataset.py        # Dataset PyTorch + transforms
│   └── models/
│       ├── config.py         # Hyperparamètres et constantes
│       ├── efficientnet.py   # Architecture EfficientNet-B0
│       ├── train.py          # Entraînement 2 phases
│       ├── evaluate.py       # Évaluation + métriques
│       └── register_model.py # Publication MLflow Registry
├── app.py                    # Interface Streamlit
├── Dockerfile                # Déploiement HF Space
└── requirements.txt

## Configuration de l'environnement

### Prérequis
- Python 3.10
- Conda (recommandé)
- Compte AWS S3
- Instance MLflow avec backend NeonDB (PostgreSQL)

### Installation

```bash
# 1. Créer l'environnement conda
conda create -n bloc6 python=3.10 -y
conda activate bloc6

# 2. Installer PyTorch CPU en premier
pip install torch==2.2.2 torchvision==0.17.2 numpy==1.26.4 \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Installer le package en mode éditable
pip install -e .
```

### Variables d'environnement

Copier `.env.example` en `.env` et renseigner les valeurs :

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `MLFLOW_TRACKING_URI` | URI du serveur MLflow (NeonDB) |
| `MLFLOW_EXPERIMENT_NAME` | Nom de l'expérience MLflow |
| `MLFLOW_MODEL_NAME` | Nom du modèle dans le registry |
| `AWS_ACCESS_KEY_ID` | Clé d'accès AWS |
| `AWS_SECRET_ACCESS_KEY` | Secret AWS |
| `AWS_DEFAULT_REGION` | Région AWS (ex: eu-west-3) |
| `S3_BUCKET` | Nom du bucket S3 |
| `DATA_PROCESSED_DIR` | Chemin vers les manifests CSV |

### Note WSL2

Sur WSL2/Ubuntu avec CPU uniquement, `NUM_WORKERS=0` est obligatoire 
dans les DataLoaders (pas de fork multiprocessing). Cette valeur est 
déjà configurée par défaut dans `src/models/config.py`.