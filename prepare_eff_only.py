"""
Script de preparation EfficientNet uniquement.
A lancer apres que les fichiers ViT sont deja generes.
"""
import os, sys, pickle
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

TRAIN_DIR  = r"C:\Users\hp\OneDrive\LungDisease_4Classes\train"
OUTPUT_DIR = r"C:\Users\hp\Desktop\newprjt\models"
EFF_MODEL_PATH = r"C:\Users\hp\Desktop\newprjt\models\model_efficientnet_b1.h5"

CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n=== Preparation EfficientNet ===")

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.spatial.distance import mahalanobis as scipy_mahal

full_model = load_model(EFF_MODEL_PATH, compile=False)
base       = EfficientNetB1(weights='imagenet', include_top=False, pooling='avg')
feat_ext   = Model(inputs=base.input, outputs=base.output)

IMG_SIZE = (240, 240)

def get_features(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = preprocess_input(arr[np.newaxis])
    return feat_ext.predict(arr, verbose=0)[0]

features_by_class = {}
for cls in CLASS_NAMES:
    cls_dir = os.path.join(TRAIN_DIR, cls)
    feats   = []
    imgs    = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for fname in tqdm(imgs, desc=cls):
        try:
            feats.append(get_features(os.path.join(cls_dir, fname)))
        except Exception:
            pass
    features_by_class[cls] = np.array(feats)
    print(f"  {cls} : {len(feats)} features")

# Seuil Energy Distance
print("\nCalcul threshold_energy...")
intra = []
for cls in CLASS_NAMES:
    F   = features_by_class[cls]
    D   = cdist(F, F, metric='euclidean')
    idx = np.triu_indices_from(D, k=1)
    intra.extend(D[idx].tolist())
threshold_energy = float(np.percentile(intra, 95))
print(f"EfficientNet -- threshold_energy (95e pct) : {threshold_energy:.4f}")

# Statistiques Mahalanobis
print("\nCalcul Mahalanobis stats...")
mahal_stats = {}
all_intra_mahal = []
for cls in CLASS_NAMES:
    F       = features_by_class[cls].astype(np.float64)
    mean    = F.mean(axis=0)
    cov     = np.cov(F, rowvar=False)
    inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
    mahal_stats[cls] = {"mean": mean, "inv_cov": inv_cov}
    for vec in F:
        d = scipy_mahal(vec, mean, inv_cov)
        all_intra_mahal.append(d)

seuil_rejet = float(np.percentile(all_intra_mahal, 95))
mahal_stats["_threshold"] = seuil_rejet
print(f"EfficientNet -- seuil_rejet Mahalanobis (95e pct) : {seuil_rejet:.4f}")

# Sauvegarde
with open(os.path.join(OUTPUT_DIR, "eff_features_by_class.pkl"), "wb") as f:
    pickle.dump(features_by_class, f)
with open(os.path.join(OUTPUT_DIR, "eff_threshold_energy.pkl"), "wb") as f:
    pickle.dump(threshold_energy, f)
with open(os.path.join(OUTPUT_DIR, "eff_mahal_stats.pkl"), "wb") as f:
    pickle.dump(mahal_stats, f)

print("\n-> eff_features_by_class.pkl sauvegarde.")
print("-> eff_threshold_energy.pkl sauvegarde.")
print("-> eff_mahal_stats.pkl sauvegarde.")
print("\nOK ! Tous les fichiers EfficientNet sont prets.")
