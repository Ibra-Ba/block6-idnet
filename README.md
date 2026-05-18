# 🛂 Bloc 6 — Détection de Fraude Documentaire

Projet de certification Bac+4 — Direction de projets de gestion de données  
**Jedha Bootcamp | Fullstack**

## Démo

👉 [ID Check Demo — Hugging Face Space](https://huggingface.co/spaces/VoxUp/id-check-demo)

## Contexte

La fraude documentaire représente un enjeu majeur pour les organisations
soumises aux obligations KYC (Know Your Customer) et aux réglementations
eIDAS. Ce projet implémente un système de détection automatique de documents
d'identité frauduleux par deep learning.

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
│   └── models/
├── space/
│   ├── inference_app/
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── README.md
│   └── mlflow_server/
│       ├── Dockerfile
│       ├── entrypoint.sh
│       ├── requirements.txt
│       └── README.md
├── data/processed/
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md




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

## Utilisation

### Entraînement
```bash
python -m src.models.train
```

### Évaluation
```bash
python -m src.models.evaluate <run_id>
```

### Publication du modèle
```bash
python -m src.models.register_model <run_id>
```

### App locale
```bash
# Depuis space/inference_app/
streamlit run app.py
```

## Source de vérité

Le dossier `space/` contient les sources de vérité pour les deux HF Spaces.
Après toute modification, mettre à jour le Space HF correspondant :

```bash
# Inference app
cp space/inference_app/* ~/fullstack-certification/bloc6/id-check-demo/

# MLflow server
cp space/mlflow_server/* <chemin_local_mlflow_server>/
```

> **Note** : Le `requirements.txt` de `space/inference_app/` ne contient
> pas `scikit-learn` et `pandas` — uniquement les dépendances nécessaires
> à l'inférence.

## Architecture globale

┌─────────────────────┐
                │   MLflow Server      │
                │   HF Space           │
                │   NeonDB + AWS S3    │
                └──────────┬──────────┘
                           │ @champion
                ┌──────────▼──────────┐
                │   ID Check Demo      │
                │   HF Space           │
                │   Streamlit          │
                └─────────────────────┘


## Relation avec le bloc4 de lead

Ce projet constitue le socle data science du système de détection.
Le projet de certification du bloc4 de lead y ajoute la couche MLOps complète : CI/CD GitHub Actions,
tests automatisés, monitoring et réentraînement continu.

