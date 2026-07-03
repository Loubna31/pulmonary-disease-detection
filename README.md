# 🫁 PneumoIA — Système de Détection des Maladies Pulmonaires

> Plateforme IA médicale de détection automatique des maladies pulmonaires avec explicabilité visuelle (XAI) et apprentissage incrémental Few-Shot  
> Développée en partenariat avec le **Service de Pneumologie — CHU Oran (Algérie)**

---

## 📌 Description

**PneumoIA** est une application web médicale complète qui permet aux médecins de soumettre des radiographies thoraciques et d'obtenir un diagnostic automatique assisté par IA, accompagné de **cartes d'explicabilité visuelle** pour justifier chaque décision du modèle.

---

## 🏥 Partenariat Hospitalier

| | |
|---|---|
| **Établissement** | CHU Oran — Centre Hospitalo-Universitaire |
| **Service** | Pneumologie |
| **Rôle** | Accès aux radiographies thoraciques réelles & validation clinique |
| **Confidentialité** | Données patients anonymisées — non incluses dans ce repository |

---

## 🎯 Maladies Détectées (4 classes)

- **COVID-19** (Corona Virus Disease)
- **Pneumonie**
- **Tuberculose**
- **Normal** (poumons sains)

---

## 🧠 Modèles IA

Le système propose **deux modèles** au choix du médecin :

### 1. Vision Transformer (ViT)
- Modèle : `google/vit-base-patch16-224-in21k` (Hugging Face Transformers)
- Détection OOD : **Energy Distance** inter-classes (seuil percentile 95)
- Explicabilité : **GradCAM++** (similarité cosinus CLS-patch sur hidden states) + **SHAP** (GradientExplainer + ViTShapWrapper)
- Carte hybride : `max(GradCAM, SHAP) + 0.4×consensus + CLAHE`
- Segmentation morphologique + overlay coloré

### 2. EfficientNet B1
- Modèle : `.h5` fine-tuné (TensorFlow / Keras)
- Détection OOD : **Energy Distance** + **Distance de Mahalanobis**
- Explicabilité : **GradCAM++** (couche `top_conv`, GradientTape) + **SHAP** (GradientExplainer)
- Carte hybride : `(0.6 × GradCAM) × (0.4 × SHAP)` + segmentation Otsu + contour cyan

---

## 🔍 Explicabilité (XAI)

Chaque prédiction génère automatiquement :
- Une **heatmap** superposée à la radiographie — zones décisives pour le diagnostic
- Un **score de sparsité** (ratio de pixels actifs)
- Un **score de focus** (concentration de l'attention du modèle)
- Une **carte de perturbation** — masquage des zones > 0.5 pour valider la robustesse du modèle

---

## 🔄 Apprentissage Incrémental (Few-Shot)

- Le médecin soumet de nouveaux exemples pour une classe rare ou inconnue
- Comparaison via **few-shot** (distance aux prototypes de classe)
- **Ré-entraînement automatique asynchrone** (threading) sans interruption du service

---

## 👥 Rôles Utilisateurs

| Rôle | Accès |
|------|-------|
| **Médecin** | Upload radiographie · Diagnostic · Explicabilité · Few-Shot · Historique |
| **Admin (Chef de service)** | Gestion des médecins · Logs de ré-entraînement · Tableau de bord |

---

## 🏗️ Architecture du Projet

```
PneumoIA/
├── app.py                  # Factory Flask
├── config.py               # Configuration
├── extensions.py           # SQLAlchemy, LoginManager
├── models.py               # User, Prediction, FewShotSample, RetrainingLog
├── run.py                  # Point d'entrée
├── ml/
│   ├── vit_model.py        # Pipeline complet ViT (prédiction + OOD + XAI)
│   └── efficient_model.py  # Pipeline complet EfficientNet B1 (prédiction + OOD + XAI)
├── models/
│   ├── vit_model_4classes_best.pth       # Poids ViT fine-tuné
│   ├── model_efficientnet_b1.h5          # Poids EfficientNet fine-tuné
│   ├── vit_features_by_class.pkl         # Features ViT par classe (OOD)
│   ├── eff_features_by_class.pkl         # Features EfficientNet par classe (OOD)
│   ├── eff_mahal_stats.pkl               # Statistiques Mahalanobis
│   └── vit_threshold.pkl / eff_threshold_energy.pkl
├── routes/
│   ├── auth.py             # Authentification
│   ├── doctor.py           # Interface médecin (upload, résultats, few-shot)
│   └── admin.py            # Tableau de bord administrateur
├── templates/              # Interface HTML
├── static/                 # CSS, JS, images
└── requirements.txt
```

---

## 🛠️ Technologies Utilisées

| Catégorie | Outils |
|-----------|--------|
| Deep Learning (ViT) | PyTorch · Hugging Face Transformers |
| Deep Learning (CNN) | TensorFlow · Keras · EfficientNet B1 |
| Explicabilité (XAI) | SHAP · GradCAM++ · GradientTape |
| Computer Vision | OpenCV · PIL · torchvision |
| Détection OOD | Energy Distance · Distance de Mahalanobis |
| Backend | Python · Flask · SQLAlchemy · Flask-Login |
| Base de données | SQLite |
| Frontend | HTML · CSS · JavaScript |

---

## ⚙️ Installation

```bash
git clone https://github.com/Loubna31/pulmonary-disease-detection.git
cd pulmonary-disease-detection
pip install -r requirements.txt
python run.py
```

---

## 🔒 Note sur les Données

Les radiographies utilisées proviennent du **service de pneumologie du CHU d'Oran** et sont confidentielles — non incluses dans ce repository.

Pour reproduire les expériences :
- [COVID-19 Radiography Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- [NIH Chest X-Ray Dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)

---

## 👩‍💻 Auteure

**Loubna Benaissa**  
Master 2 — Aide à la Décision et Systèmes Intelligents  
📧 benaissaloubna2@gmail.com

---

## 📄 Licence

MIT License — voir le fichier [LICENSE](LICENSE).
