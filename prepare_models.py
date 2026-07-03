"""




Ce script calcule et sauvegarde dans newprjt/models/ :
  - vit_features_by_class.pkl
  - vit_threshold.pkl
  - eff_features_by_class.pkl
  - eff_threshold_energy.pkl
  - eff_mahal_stats.pkl

CONFIGURATION — mets à jour les chemins ci-dessous avant de lancer.
"""

import os, pickle
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

# ─── À CONFIGURER ───────────────────────────────────────────────────────────
TRAIN_DIR = r"C:\Users\hp\OneDrive\LungDisease_4Classes\train"
OUTPUT_DIR = r"C:\Users\hp\Desktop\newprjt\models"

VIT_MODEL_PATH = r"C:\Users\hp\Desktop\newprjt\models\vit_model_4classes_best.pth"
EFF_MODEL_PATH = r"C:\Users\hp\Desktop\newprjt\models\model_efficientnet_b1.h5"
# ────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def energy_distance(X, Y):
    d_xy = cdist(X, Y, metric='euclidean').mean()
    d_xx = cdist(X, X, metric='euclidean').mean() if len(X) > 1 else 0.0
    d_yy = cdist(Y, Y, metric='euclidean').mean() if len(Y) > 1 else 0.0
    return float(2 * d_xy - d_xx - d_yy)


# ════════════════════════════════════════════════════════════════════════════
# ViT
# ════════════════════════════════════════════════════════════════════════════

def prepare_vit():
    print("\n=== Préparation ViT ===")
    import torch
    from transformers import ViTForImageClassification, ViTConfig, AutoImageProcessor
    from PIL import Image

    MODEL_NAME = "google/vit-base-patch16-224-in21k"
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charge uniquement la config (déjà en cache, ~1KB) — évite le téléchargement des poids HF (~330MB)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    config    = ViTConfig.from_pretrained(MODEL_NAME)
    config.num_labels = len(CLASS_NAMES)
    model     = ViTForImageClassification(config)   # initialisation aléatoire
    state     = torch.load(VIT_MODEL_PATH, map_location=device)
    model.load_state_dict(state, strict=False)      # charge nos poids fine-tunés
    model.to(device)
    model.eval()

    features_by_class = {}
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        feats   = []
        imgs    = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        for fname in tqdm(imgs, desc=cls):
            img_path = os.path.join(cls_dir, fname)
            try:
                img    = Image.open(img_path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True)
                cls_tok = out.hidden_states[-1][:, 0, :].cpu().numpy()[0]  # (768,)
                feats.append(cls_tok)
            except Exception:
                pass
        features_by_class[cls] = np.array(feats)
        print(f"  {cls} : {len(feats)} features")

    # Seuil OOD — 95e percentile des Energy Distances (point vs classe)
    # Optimisation O(N²) par classe au lieu de O(N³) : d_yy calculé une seule fois
    print("\nCalcul du seuil OOD ViT (distances intra-classe)...")
    intra = []
    for cls in CLASS_NAMES:
        F   = features_by_class[cls].astype(np.float64)
        D   = cdist(F, F, metric='euclidean')          # (N,N) — une seule fois
        d_xy_per_pt = D.mean(axis=1)                   # vecteur (N,) — mean dist de chaque pt vers le reste
        d_yy        = D.mean()                          # scalaire — mean pairwise distance globale
        eds = 2.0 * d_xy_per_pt - d_yy                # ED approx pour chaque point
        intra.extend(eds.tolist())
        print(f"  {cls} : {len(eds)} distances calculées")
    FINAL_THRESHOLD = float(np.percentile(intra, 95))
    print(f"\nViT — FINAL_THRESHOLD (95e pct) : {FINAL_THRESHOLD:.4f}")

    with open(os.path.join(OUTPUT_DIR, "vit_features_by_class.pkl"), "wb") as f:
        pickle.dump(features_by_class, f)
    with open(os.path.join(OUTPUT_DIR, "vit_threshold.pkl"), "wb") as f:
        pickle.dump(FINAL_THRESHOLD, f)
    print("-> vit_features_by_class.pkl et vit_threshold.pkl sauvegardés.")


# ════════════════════════════════════════════════════════════════════════════
# EfficientNet
# ════════════════════════════════════════════════════════════════════════════

def prepare_efficient():
    print("\n=== Préparation EfficientNet ===")
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
        img  = load_img(img_path, target_size=IMG_SIZE)
        arr  = img_to_array(img)
        arr  = preprocess_input(arr[np.newaxis])
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

    # Seuil Energy Distance — 95e percentile intra-classe
    intra = []
    for cls in CLASS_NAMES:
        F   = features_by_class[cls]
        D   = cdist(F, F, metric='euclidean')
        idx = np.triu_indices_from(D, k=1)
        intra.extend(D[idx].tolist())
    threshold_energy = float(np.percentile(intra, 95))
    print(f"\nEfficientNet — threshold_energy (95e pct) : {threshold_energy:.4f}")

    # Statistiques Mahalanobis par classe
    mahal_stats = {}
    all_intra_mahal = []
    for cls in CLASS_NAMES:
        F      = features_by_class[cls].astype(np.float64)
        mean   = F.mean(axis=0)
        cov    = np.cov(F, rowvar=False)
        inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
        mahal_stats[cls] = {"mean": mean, "inv_cov": inv_cov}
        for vec in F:
            d = scipy_mahal(vec, mean, inv_cov)
            all_intra_mahal.append(d)

    seuil_rejet = float(np.percentile(all_intra_mahal, 95))
    mahal_stats["_threshold"] = seuil_rejet
    print(f"EfficientNet — seuil_rejet Mahalanobis (95e pct) : {seuil_rejet:.4f}")

    with open(os.path.join(OUTPUT_DIR, "eff_features_by_class.pkl"), "wb") as f:
        pickle.dump(features_by_class, f)
    with open(os.path.join(OUTPUT_DIR, "eff_threshold_energy.pkl"), "wb") as f:
        pickle.dump(threshold_energy, f)
    with open(os.path.join(OUTPUT_DIR, "eff_mahal_stats.pkl"), "wb") as f:
        pickle.dump(mahal_stats, f)
    print("-> eff_features_by_class.pkl, eff_threshold_energy.pkl, eff_mahal_stats.pkl sauvegardés.")


if __name__ == "__main__":
    print("Étape 1 : Préparation ViT")
    prepare_vit()
    print("\nÉtape 2 : Préparation EfficientNet")
    prepare_efficient()
    print("\n✅ Tous les fichiers .pkl sont prêts dans newprjt/models/")
