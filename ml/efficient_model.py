"""
=============================================================================
 ml/efficient_model.py
 Pipeline complet EfficientNetB1 (TensorFlow/Keras) pour PneumoIA.

 Étapes du pipeline :
   1. Prédiction  : modèle .h5 entraîné (4 classes originales)
   2. Features    : EfficientNetB1(include_top=False, pooling='avg') → (1280,)
   3. OOD Energy  : Energy Distance vs classes, seuil = percentile 10 intra-classe
   4. OOD Mahal   : scipy.spatial.distance.mahalanobis, seuil = percentile 90
   5. GradCAM++   : GradientTape + pondération alpha (couche top_conv)
   6. SHAP        : GradientExplainer + background = vraies images train
   7. Hybrid      : (0.6 × gradcam) × (0.4 × shap) + normalisation
   8. Segmentation: Otsu + overlay coloré (HOT) + contour cyan
   9. Perturbation: masquage pixels > 0.5, intensity = 0.8
  10. Scores      : sparsity (ratio pixels actifs), focus (concentration)

 """

# ─── Bibliothèques standard Python ───────────────────────────────────────────
import os        # manipulation des chemins de fichiers
import uuid      # noms de fichiers uniques pour les cartes d'explicabilité
import shutil    # copie de fichiers (few-shot)
import pickle    # sérialisation/désérialisation des features et statistiques
import warnings  # suppression des warnings SHAP non critiques
import random    # valeurs aléatoires pour le mode stub

# ─── Calcul scientifique ──────────────────────────────────────────────────────
import numpy as np                                           # opérations matricielles
import cv2                                                   # traitement d'images (resize, overlay)
from scipy.spatial.distance import cdist                    # distances entre ensembles de vecteurs
from scipy.spatial.distance import mahalanobis as sp_mahal  # distance de Mahalanobis

# ─── TensorFlow / Keras ───────────────────────────────────────────────────────
import tensorflow as tf                                              # GradientTape pour GradCAM++
from tensorflow.keras.models import load_model, Model               # chargement modèle .h5 + extracteur features
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB1,      # backbone extracteur de features (1280-d)
    preprocess_input,    # prétraitement spécifique à EfficientNetB1
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array   # chargement des images

# ─── Explicabilité SHAP ───────────────────────────────────────────────────────
import shap   # SHapley Additive exPlanations (GradientExplainer)

# =============================================================================
#  Configuration — CHEMINS À METTRE À JOUR
# =============================================================================
MODEL_PATH            = r"C:\Users\hp\Desktop\newprjt\models\model_efficientnet_b1.h5"
FEATURES_PATH         = r"C:\Users\hp\Desktop\newprjt\models\eff_features_by_class.pkl"
THRESHOLD_ENERGY_PATH = r"C:\Users\hp\Desktop\newprjt\models\eff_threshold_energy.pkl"
MAHAL_STATS_PATH      = r"C:\Users\hp\Desktop\newprjt\models\eff_mahal_stats.pkl"
TRAIN_DIR             = r"C:\Users\hp\OneDrive\LungDisease_4Classes\train"   # images fond SHAP
LAST_CONV_LAYER       = "top_conv"   # dernière couche convolutive pour GradCAM++
NUM_BACKGROUND        = 20           # nombre d'images de fond pour SHAP

# =============================================================================
#  Constantes du modèle
# =============================================================================
ORIGINAL_CLASSES = ['Corona Virus Disease', 'Normal', 'Pneumonia', 'Tuberculosis']
CLASS_NAMES      = list(ORIGINAL_CLASSES)   # mis à jour dynamiquement après ré-entraînement
IMG_SIZE         = 240                       # taille des images en entrée (EfficientNetB1)

# Répertoire de sortie pour les cartes d'explicabilité (défini par set_explainability_dir)
EXPLAINABILITY_DIR = None

# =============================================================================
#  Singletons — chargés une seule fois (lazy loading)
# =============================================================================
_model             = None    # modèle Keras entraîné (classification 4 classes)
_feat_extractor    = None    # extracteur features EfficientNetB1 (1280-d)
_features_by_class = None    # dict {classe: np.ndarray (N, 1280)} — features train
_threshold_energy  = None    # seuil Energy Distance (percentile 10 intra-classe)
_mahal_stats       = None    # dict {classe: {'mean':..., 'inv_cov':...}, '_threshold': float}
_background_data   = None    # array numpy (N, 240, 240, 3) — images fond SHAP

# Cache des features few-shot par classe — évite le recalcul à chaque comparaison
_few_shot_cache_eff = {}   # {cls_name:count → np.ndarray (N, 1280)}


# =============================================================================
#  API publique — Configuration du répertoire de sortie
# =============================================================================
def set_explainability_dir(path: str):
    """
    Définit le répertoire où les cartes d'explicabilité seront sauvegardées.
    Doit être appelé avant toute génération de cartes.
    """
    global EXPLAINABILITY_DIR
    EXPLAINABILITY_DIR = path


# =============================================================================
#  Vérifications de disponibilité des fichiers
# =============================================================================
def _ready_energy() -> bool:
    """Vérifie que les fichiers nécessaires pour la méthode Energy existent."""
    return all(os.path.exists(p) for p in [MODEL_PATH, FEATURES_PATH, THRESHOLD_ENERGY_PATH])

def _ready_mahal() -> bool:
    """Vérifie que les fichiers nécessaires pour la méthode Mahalanobis existent."""
    return all(os.path.exists(p) for p in [MODEL_PATH, MAHAL_STATS_PATH])


# =============================================================================
#  Chargement des modèles et données (lazy)
# =============================================================================
def _load_model():
    """
    Charge le modèle Keras (.h5) et crée l'extracteur de features.
    Utilise un pattern singleton : ne charge qu'une seule fois par processus.
    """
    global _model, _feat_extractor

    if _model is not None:
        return   # déjà chargé

    # Modèle de classification EfficientNetB1 entraîné
    _model = load_model(MODEL_PATH, compile=False)
    _model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Extracteur de features : EfficientNetB1 pré-entraîné ImageNet,
    # sans la tête de classification, avec Global Average Pooling → (1, 1280)
    base            = EfficientNetB1(weights='imagenet', include_top=False, pooling='avg')
    _feat_extractor = Model(inputs=base.input, outputs=base.output)


def _load_energy_data():
    """
    Charge les features de référence et le seuil Energy Distance.
    Singleton : ne charge qu'une seule fois.
    """
    global _features_by_class, _threshold_energy

    if _features_by_class is not None:
        return   # déjà chargé

    with open(FEATURES_PATH, "rb") as f:
        _features_by_class = pickle.load(f)
    with open(THRESHOLD_ENERGY_PATH, "rb") as f:
        _threshold_energy = pickle.load(f)


def _load_mahal_data():
    """
    Charge les statistiques Mahalanobis (mean + inv_cov) pour chaque classe.
    Singleton : ne charge qu'une seule fois.
    """
    global _mahal_stats

    if _mahal_stats is not None:
        return   # déjà chargé

    with open(MAHAL_STATS_PATH, "rb") as f:
        _mahal_stats = pickle.load(f)


def _load_background():
    """
    Charge les images du dataset d'entraînement comme background SHAP.
    Utilise de vraies images radiologiques pour un background représentatif.
    Fallback : bruit gaussien si le dossier train est introuvable.

    Retourne : array numpy (N, 240, 240, 3) preprocessé pour EfficientNetB1
    """
    global _background_data

    if _background_data is not None:
        return _background_data   # déjà chargé

    images = []
    if os.path.exists(TRAIN_DIR):
        for cls in CLASS_NAMES:
            cls_path = os.path.join(TRAIN_DIR, cls)
            if not os.path.exists(cls_path):
                continue
            for fname in os.listdir(cls_path):
                img_path = os.path.join(cls_path, fname)
                try:
                    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                    arr = img_to_array(img)
                    arr = preprocess_input(arr)
                    images.append(arr)
                    if len(images) >= NUM_BACKGROUND:
                        break
                except Exception:
                    continue
            if len(images) >= NUM_BACKGROUND:
                break

    if images:
        _background_data = np.array(images[:NUM_BACKGROUND], dtype=np.float32)
    else:
        # Fallback : bruit gaussien centré (approxime la distribution prétraitée)
        print("[EfficientNet SHAP] Dossier train introuvable → bruit gaussien comme background")
        _background_data = np.random.normal(0, 1, (NUM_BACKGROUND, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)

    return _background_data


# =============================================================================
#  Prétraitement des images
# =============================================================================
def _preprocess(image_path: str):
    """
    Charge et prétraite une image pour EfficientNetB1.

    Retourne :
        img_preprocessed : array (1, 240, 240, 3) prêt pour le modèle
        img_np           : array (240, 240, 3) float [0, 1] pour l'overlay
    """
    img    = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr    = img_to_array(img)
    img_np = arr.astype(np.float32) / 255.0   # version normalisée pour l'affichage
    arr_p  = preprocess_input(arr.copy())      # prétraitement EfficientNet (standardisation)
    return np.expand_dims(arr_p, axis=0), img_np


def _extract_features(img_array) -> np.ndarray:
    """
    Extrait le vecteur de features EfficientNetB1 (1280-d).
    Paramètre : img_array de forme (1, 240, 240, 3) prétraité
    Retourne  : array (1, 1280)
    """
    return _feat_extractor.predict(img_array, verbose=0).reshape(1, -1)


# =============================================================================
#  Energy Distance (OOD)
# =============================================================================
def _energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Calcule l'Energy Distance symétrique entre deux ensembles de points X et Y.
    Formule : ED = 2·E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]

    Retourne float('inf') si l'un des ensembles est vide.
    """
    if len(X) == 0 or len(Y) == 0:
        return float('inf')
    d_xy = cdist(X, Y, metric='euclidean').mean()
    d_xx = cdist(X, X, metric='euclidean').mean()
    d_yy = cdist(Y, Y, metric='euclidean').mean()
    return float(2 * d_xy - d_xx - d_yy)


# =============================================================================
#  Prédiction OOD par Mahalanobis
# =============================================================================
def _predict_ood_mahal(features: np.ndarray):
    """
    Calcule la distance de Mahalanobis entre le vecteur de features et
    chaque classe (originales + nouvelles) pour identifier la plus proche.

    Retourne : (best_class, best_distance)
    """
    best_class = None
    best_dist  = float("inf")

    for cls, stats in _mahal_stats.items():
        if cls == "_threshold":   # clé spéciale contenant le seuil global
            continue
        try:
            dist = float(sp_mahal(features, stats["mean"], stats["inv_cov"]))
        except Exception:
            continue
        if dist < best_dist:
            best_dist  = dist
            best_class = cls

    return best_class, best_dist


# =============================================================================
#  GradCAM++ (Keras)
# =============================================================================
def _compute_gradcam(img_array, pred_idx: int) -> np.ndarray:
    """
    Calcule la carte GradCAM++ pour EfficientNetB1.
    Utilise la couche LAST_CONV_LAYER (top_conv) comme cible.
    Applique la pondération alpha de GradCAM++ (alpha = numerator/denominator).

    Paramètres :
        img_array : tensor d'image prétraité (1, 240, 240, 3)
        pred_idx  : indice de la classe à expliquer

    Retourne : heatmap float32 (240, 240) dans [0, 1]
    """
    # Modèle intermédiaire : sortie de top_conv + sortie finale
    grad_model = tf.keras.models.Model(
        _model.inputs,
        [_model.get_layer(LAST_CONV_LAYER).output, _model.output]
    )

    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_tensor)
        loss = preds[:, pred_idx]

    grads = tape.gradient(loss, conv_output)

    # Pondération GradCAM++ : alpha = gradients² / (2·gradients² + activations·gradients³)
    numerator   = grads ** 2
    denominator = (
        2 * grads ** 2
        + tf.reduce_sum(conv_output * grads ** 3, axis=(0, 1, 2))
    )
    denominator = tf.where(denominator != 0, denominator, 1e-8)
    alpha       = numerator / denominator
    weights     = tf.reduce_sum(alpha * tf.maximum(grads, 0), axis=(0, 1, 2))

    # Combinaison linéaire pondérée des feature maps
    cam = tf.reduce_sum(weights * conv_output[0], axis=-1)
    cam = tf.maximum(cam, 0)                          # ReLU
    cam = cam / (tf.reduce_max(cam) + 1e-8)           # normalisation [0, 1]
    cam = cam.numpy()

    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    return cam.astype(np.float32)


# =============================================================================
#  SHAP (EfficientNet)
# =============================================================================
def _get_shap_map(img_array, pred_idx: int, background) -> np.ndarray:
    """
    Calcule la carte SHAP via GradientExplainer sur le modèle EfficientNetB1.
    Applique GaussianBlur et seuillage au 75e percentile (clean_shap).

    Paramètres :
        img_array  : tensor prétraité (1, 240, 240, 3)
        pred_idx   : indice de la classe à expliquer
        background : array (N, 240, 240, 3) — images de fond pour SHAP

    Retourne : heatmap float32 (240, 240) dans [0, 1]
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # supprimer les warnings SHAP non critiques
        explainer   = shap.GradientExplainer(_model, background)
        shap_values = explainer.shap_values(img_array)

    # Extraction des valeurs SHAP pour la classe prédite
    if isinstance(shap_values, list):
        shap_map = shap_values[pred_idx][0]
    else:
        shap_map = shap_values[0]

    # Réduction des dimensions spatiales (H, W) en moyennant sur les canaux
    while len(shap_map.shape) > 2:
        shap_map = np.mean(np.abs(shap_map), axis=-1)

    # Normalisation + clean_shap : lissage gaussien + seuillage 75e percentile
    shap_map = shap_map / (shap_map.max() + 1e-8)
    shap_map = cv2.GaussianBlur(shap_map.astype(np.float32), (15, 15), 0)
    thresh   = np.percentile(shap_map, 75)
    shap_map[shap_map < thresh] = 0   # supprimer les contributions non significatives

    return (shap_map / (shap_map.max() + 1e-8)).astype(np.float32)


# =============================================================================
#  Fusion Hybrid (EfficientNet)
# =============================================================================
def _hybrid_fusion(gradcam: np.ndarray, shap_map: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    Fusion hybride EfficientNet :
        hybrid = (alpha × gradcam) × ((1-alpha) × shap_map)
    Normalisation finale par le maximum.

    Formule issue du notebook prjtpfe.ipynb.
    Retourne : heatmap float32 (240, 240) dans [0, 1]
    """
    gc     = cv2.resize(gradcam.astype(np.float32), (IMG_SIZE, IMG_SIZE))
    hybrid = (alpha * gc) * ((1 - alpha) * shap_map)
    return (hybrid / (hybrid.max() + 1e-8)).astype(np.float32)


# =============================================================================
#  Segmentation Otsu (EfficientNet)
# =============================================================================
def _otsu_segment(hybrid: np.ndarray, img_np: np.ndarray = None) -> np.ndarray:
    """
    Segmentation par seuillage d'Otsu sur la carte hybride.
    Produit un overlay coloré (HOT) sur l'image originale avec contour cyan.

    Paramètres :
        hybrid : heatmap hybride float32 (H, W)
        img_np : image originale float32 (H, W, 3) — pour l'overlay

    Retourne :
        - Image BGR avec overlay si img_np fourni
        - Masque vert sur fond noir sinon
    """
    uint8   = np.uint8(255 * hybrid)
    _, thresh = cv2.threshold(uint8, 0, 255, cv2.THRESH_OTSU)

    if img_np is not None:
        # ── Overlay coloré sur l'image originale ──────────────────────────────
        orig    = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        mask    = thresh > 0

        # Fond assombri + zones actives en HOT
        result  = (orig * 0.35).astype(np.uint8)
        heat    = cv2.applyColorMap(np.uint8(255 * hybrid), cv2.COLORMAP_HOT)
        blended = cv2.addWeighted(orig, 0.35, heat, 0.65, 0)
        result[mask] = blended[mask]

        # Contour cyan
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 255), 2)
        return result

    # ── Fallback : masque vert sur fond noir ──────────────────────────────────
    colored = np.zeros((*thresh.shape, 3), dtype=np.uint8)
    colored[thresh > 0] = [0, 200, 80]
    return colored


# =============================================================================
#  Validation par perturbation
# =============================================================================
def _perturb_drop(img_array, heatmap: np.ndarray,
                  pred_idx: int, orig_conf: float, intensity: float = 0.8) -> float:
    """
    Mesure la chute de confiance après masquage des zones actives.
    Principe : si le masquage fait chuter la confiance de façon importante (> 20%),
    les zones identifiées sont bien responsables de la décision.

    Paramètres :
        img_array  : array numpy (1, 240, 240, 3) prétraité
        heatmap    : carte d'activation float32 (H, W)
        pred_idx   : indice de la classe prédite
        orig_conf  : confiance originale en % (0-100)
        intensity  : intensité du masquage (0.8 = 80% d'obscurcissement)

    Retourne : chute de confiance en % (valeur positive = baisse)
    """
    h     = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    h     = h / (h.max() + 1e-8)
    mask  = h > 0.5             # masque des zones actives

    # Masquer les zones actives dans l'image
    img_c = img_array[0].copy()
    img_c[mask] = img_c[mask] * (1 - intensity)

    preds  = _model.predict(np.expand_dims(img_c, 0), verbose=0)[0]
    p_conf = float(preds[pred_idx]) * 100
    return round(orig_conf - p_conf, 2)


# =============================================================================
#  Scores qualitatifs
# =============================================================================
def _sparsity_score(m) -> float:
    """Ratio de pixels actifs — faible = activation focalisée."""
    return float(np.sum(m > 0) / m.size)

def _focus_score(m) -> float:
    """Rapport max/moyenne — élevé = pic d'activation concentré."""
    return float(np.max(m) / (np.mean(m) + 1e-8))


# =============================================================================
#  Sauvegarde des cartes (utilitaires)
# =============================================================================
def _save(arr_bgr: np.ndarray, prefix: str) -> str:
    """Sauvegarde une image BGR dans EXPLAINABILITY_DIR avec un nom unique."""
    fname = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    cv2.imwrite(os.path.join(EXPLAINABILITY_DIR, fname), arr_bgr)
    return fname

def _overlay(img_np: np.ndarray, heatmap: np.ndarray,
             cmap=cv2.COLORMAP_JET, alpha_img: float = 0.6) -> np.ndarray:
    """Superpose une heatmap colorée sur l'image originale avec alpha_img."""
    orig    = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cmap)
    return cv2.addWeighted(orig, alpha_img, colored, 1 - alpha_img, 0)


# =============================================================================
#  Mode dégradé (stub)
# =============================================================================
def _stub(label: str) -> dict:
    """
    Retourne des données aléatoires quand les fichiers modèle sont manquants.
    Affiche un message pour informer qu'il faut lancer prepare_models.py.
    """
    print(f"[EfficientNet/{label}] STUB — fichiers pkl manquants. Lancez prepare_models.py")
    is_k = random.random() > 0.25
    d    = round(random.uniform(8.0, 25.0), 4)
    thr  = 16.80 if label == "energy" else 12.0
    return {
        'disease'              : CLASS_NAMES[random.randint(0, 3)] if is_k else None,
        'confidence'           : round(random.uniform(70, 99), 2) if is_k else None,
        'is_known'             : d < thr,
        'ood_distance'         : d,
        'ood_threshold'        : thr,
        'gradcam_filename'     : None,
        'shap_filename'        : None,
        'hybrid_filename'      : None,
        'segmentation_filename': None,
        'perturb_drop_gradcam' : round(random.uniform(15, 65), 2),
        'perturb_drop_shap'    : round(random.uniform(10, 55), 2),
        'perturb_drop_hybrid'  : round(random.uniform(25, 75), 2),
        'sparsity_score'       : round(random.uniform(0.3, 0.9), 3),
        'focus_score'          : round(random.uniform(0.4, 0.95), 3),
    }


# =============================================================================
#  Pipeline explicabilité partagé (interne)
# =============================================================================
def _run_explainability(img_array, img_np: np.ndarray,
                        pred_idx: int, confidence: float) -> dict:
    """
    Exécute le pipeline complet d'explicabilité :
    GradCAM++ → SHAP → Hybrid → Segmentation → Perturbation → Scores.

    Partagé par predict_energy, predict_mahalanobis et compute_explainability.

    Retourne un dict avec les clés :
        gradcam_filename, shap_filename, hybrid_filename, segmentation_filename,
        perturb_drop_gradcam, perturb_drop_shap, perturb_drop_hybrid,
        sparsity_score, focus_score
    """
    background = _load_background()

    # ── GradCAM++ ─────────────────────────────────────────────────────────────
    gradcam_map = _compute_gradcam(img_array, pred_idx)
    gc_fname    = _save(_overlay(img_np, gradcam_map), "gradcam_eff")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    try:
        shap_map = _get_shap_map(img_array, pred_idx, background)
        sh_fname = _save(_overlay(img_np, shap_map, cmap=cv2.COLORMAP_VIRIDIS), "shap_eff")
    except Exception as e:
        print(f"[EfficientNet SHAP] Erreur : {e} — fallback sur GradCAM++")
        shap_map = gradcam_map.copy()
        sh_fname = None

    # ── Hybrid ────────────────────────────────────────────────────────────────
    hybrid_map = _hybrid_fusion(gradcam_map, shap_map)
    hyb_fname  = _save(_overlay(img_np, hybrid_map, cmap=cv2.COLORMAP_HOT), "hybrid_eff")

    # ── Segmentation Otsu ─────────────────────────────────────────────────────
    seg_bgr   = _otsu_segment(hybrid_map, img_np)
    seg_fname = _save(seg_bgr, "seg_eff")

    # ── Validation par perturbation ───────────────────────────────────────────
    drop_gc  = _perturb_drop(img_array, gradcam_map, pred_idx, confidence)
    drop_sh  = _perturb_drop(img_array, shap_map,    pred_idx, confidence)
    drop_hyb = _perturb_drop(img_array, hybrid_map,  pred_idx, confidence)

    return {
        'gradcam_filename'     : gc_fname,
        'shap_filename'        : sh_fname,
        'hybrid_filename'      : hyb_fname,
        'segmentation_filename': seg_fname,
        'perturb_drop_gradcam' : drop_gc,
        'perturb_drop_shap'    : drop_sh,
        'perturb_drop_hybrid'  : drop_hyb,
        'sparsity_score'       : round(_sparsity_score(hybrid_map), 4),
        'focus_score'          : round(_focus_score(hybrid_map), 4),
    }


# =============================================================================
#  API publique — Prédiction EfficientNet + Energy Distance
# =============================================================================
def predict_energy(image_path: str) -> dict:
    """
    Prédiction de maladie + détection OOD par Energy Distance.
    Détecte les cas de toutes les classes connues (originales + few-shot).

    Retourne un dict avec les clés :
        disease, confidence (%), is_known, ood_distance, ood_threshold,
        _pred_idx (interne), _confidence (interne)
    """
    if not _ready_energy():
        return _stub("energy")

    _load_model()
    _load_energy_data()

    img_array, _ = _preprocess(image_path)

    # Softmax sur les 4 classes originales
    preds     = _model.predict(img_array, verbose=0)[0]
    orig_idx  = int(np.argmax(preds))
    orig_conf = float(preds[orig_idx] * 100)

    features = _extract_features(img_array)   # (1, 1280)

    # ── Energy Distance vers toutes les classes ───────────────────────────────
    all_cls   = list(_features_by_class.keys())
    distances = {
        cls: _energy_distance(features, _features_by_class[cls])
        for cls in all_cls
    }
    best_cls = min(distances, key=distances.get)
    best_ed  = distances[best_cls]
    is_known = best_ed < float(_threshold_energy)

    # ── Calcul de la confiance ─────────────────────────────────────────────────
    if is_known:
        if best_cls in ORIGINAL_CLASSES:
            orig_softmax_idx = ORIGINAL_CLASSES.index(best_cls)
            confidence = float(preds[orig_softmax_idx] * 100) if orig_softmax_idx < len(preds) else None
        else:
            # Nouvelle classe few-shot → score de proximité normalisé
            thr        = float(_threshold_energy)
            confidence = round(max(0.0, min(100.0, (1.0 - best_ed / thr) * 100)), 2) if thr > 0 else None
    else:
        confidence = None   # OOD → pas de confiance fiable

    return {
        'disease'      : best_cls if is_known else None,
        'confidence'   : confidence,
        'is_known'     : is_known,
        'ood_distance' : round(best_ed, 6),
        'ood_threshold': round(float(_threshold_energy), 6),
        '_pred_idx'    : orig_idx,      # utilisé en interne pour l'explicabilité
        '_confidence'  : round(orig_conf, 2),
    }


# =============================================================================
#  API publique — Prédiction EfficientNet + Mahalanobis
# =============================================================================
def predict_mahalanobis(image_path: str) -> dict:
    """
    Prédiction de maladie + détection OOD par distance de Mahalanobis.
    Détecte les cas de toutes les classes connues (originales + few-shot).

    Retourne un dict avec les clés :
        disease, confidence (%), is_known, ood_distance, ood_threshold,
        _pred_idx (interne), _confidence (interne)
    """
    if not _ready_mahal():
        return _stub("mahalanobis")

    _load_model()
    _load_mahal_data()

    img_array, _ = _preprocess(image_path)

    # Softmax sur les 4 classes originales
    preds     = _model.predict(img_array, verbose=0)[0]
    orig_idx  = int(np.argmax(preds))
    orig_conf = float(preds[orig_idx] * 100)

    features        = _extract_features(img_array)[0]   # (1280,)
    best_cls, best_dist = _predict_ood_mahal(features)
    seuil_rejet     = float(_mahal_stats.get("_threshold", 15.0))
    is_known        = best_dist <= seuil_rejet

    # ── Calcul de la confiance ─────────────────────────────────────────────────
    if is_known:
        if best_cls in ORIGINAL_CLASSES:
            orig_softmax_idx = ORIGINAL_CLASSES.index(best_cls)
            confidence = float(preds[orig_softmax_idx] * 100) if orig_softmax_idx < len(preds) else None
        else:
            confidence = round(max(0.0, min(100.0, (1.0 - best_dist / seuil_rejet) * 100)), 2) if seuil_rejet > 0 else None
    else:
        confidence = None

    return {
        'disease'      : best_cls if is_known else None,
        'confidence'   : confidence,
        'is_known'     : is_known,
        'ood_distance' : round(best_dist, 6),
        'ood_threshold': round(seuil_rejet, 6),
        '_pred_idx'    : orig_idx,
        '_confidence'  : round(orig_conf, 2),
    }


# =============================================================================
#  API publique — Explicabilité à la demande
# =============================================================================
def compute_explainability(image_path: str, pred_idx: int = None) -> dict:
    """
    Calcule les cartes d'explicabilité (GradCAM++, SHAP, Hybrid, Segmentation)
    et les métriques de perturbation, à la demande du médecin.

    Paramètres :
        image_path : chemin absolu vers l'image radiologique
        pred_idx   : indice de la classe (None = classe la plus probable)

    Retourne le même dict que _run_explainability()
    """
    _load_model()
    img_array, img_np = _preprocess(image_path)

    if pred_idx is None:
        preds    = _model.predict(img_array, verbose=0)[0]
        pred_idx = int(np.argmax(preds))

    confidence = float(_model.predict(img_array, verbose=0)[0][pred_idx] * 100)

    return _run_explainability(img_array, img_np, pred_idx, confidence)


# =============================================================================
#  API publique — Comparaison few-shot (EfficientNet)
# =============================================================================
def _extract_eff_feature(image_path: str) -> np.ndarray:
    """
    Extrait le vecteur de features EfficientNetB1 (1280,) d'une image.
    Retourne un array numpy (1280,) en float64.
    """
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img)
    arr = preprocess_input(arr[np.newaxis])
    return _feat_extractor.predict(arr, verbose=0)[0].astype(np.float64)   # (1280,)


def _get_eff_class_features(cls_path: str, cls_name: str) -> np.ndarray:
    """
    Retourne la matrice de features (N, 1280) pour une classe few-shot.
    Utilise un cache en mémoire indexé par (nom_classe, nombre_images).
    """
    imgs = sorted(
        f for f in os.listdir(cls_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if not imgs:
        return None

    cache_key = f"{cls_name}:{len(imgs)}"
    if cache_key in _few_shot_cache_eff:
        return _few_shot_cache_eff[cache_key]

    feats = []
    for fname in imgs:
        try:
            feats.append(_extract_eff_feature(os.path.join(cls_path, fname)))
        except Exception:
            continue

    if not feats:
        return None

    arr                          = np.array(feats, dtype=np.float64)   # (N, 1280)
    _few_shot_cache_eff[cache_key] = arr
    return arr


def _mahal_dist_and_threshold(query: np.ndarray, feats: np.ndarray,
                               percentile: int = 95, reg: float = 1e-4):
    """
    Calcule la distance de Mahalanobis entre query et la distribution de la classe,
    ainsi que le seuil intra-classe au percentile donné.

    Stratégie adaptative selon le nombre d'échantillons :
        N < 2                : distance euclidienne, seuil = inf
        N < max(10, d/4)     : covariance diagonale (régime few-shot)
        Sinon                : covariance complète régularisée

    Retourne : (distance, threshold_intra_classe)
    """
    n, d   = feats.shape
    mean   = feats.mean(axis=0)

    if n < 2:
        return float(np.linalg.norm(query - mean)), float('inf')

    if n < max(10, d // 4):
        # Covariance diagonale — évite la singularité en régime few-shot
        var     = feats.var(axis=0) + reg
        inv_cov = np.diag(1.0 / var)
    else:
        # Covariance complète régularisée
        cov = np.cov(feats, rowvar=False) + reg * np.eye(d)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Matrice singulière → fallback diagonal
            inv_cov = np.diag(1.0 / (np.diag(cov) + reg))

    dist  = float(sp_mahal(query, mean, inv_cov))

    # Seuil adaptatif selon le nombre d'échantillons disponibles :
    #   N ≤ 5  : très peu d'images → seuil = moyenne + 3×écart-type (règle des 3-sigma)
    #            évite que le percentile 95 sur 4-5 valeurs soit trop serré
    #   N < 20 : peu d'images → percentile 99 (plus permissif que 95)
    #   N ≥ 20 : assez d'images → percentile 95 (comportement standard)
    intra = [float(sp_mahal(f, mean, inv_cov)) for f in feats]
    arr   = np.array(intra)

    if n <= 5:
        std = float(arr.std()) if arr.std() > 0 else float(arr.mean() * 0.5)
        thr = float(arr.mean() + 3.0 * std)
    elif n < 20:
        thr = float(np.percentile(arr, 99))
    else:
        thr = float(np.percentile(arr, percentile))

    return dist, thr


def few_shot_compare(image_path: str, few_shot_dir: str, **kwargs) -> dict:
    """
    Compare l'image avec la base few-shot via distance de Mahalanobis
    sur les features EfficientNetB1 (1280 dimensions).
    Valable pour les méthodes Energy ET Mahalanobis (même extracteur).

    Retourne un dict avec les clés :
        found     : bool — True si une classe proche a été identifiée
        disease   : str|None — nom de la classe identifiée
        distance  : float — distance de Mahalanobis vers la meilleure classe
        threshold : float — seuil utilisé pour la décision
        counts    : dict {classe: nb_images}
    """
    _load_model()
    if not os.path.exists(few_shot_dir):
        return {'found': False, 'disease': None, 'distance': 0.0, 'threshold': 0.0, 'counts': {}}

    try:
        img_array, _ = _preprocess(image_path)
        query_feat   = _extract_features(img_array)[0].astype(np.float64)   # (1280,)
    except Exception as e:
        return {
            'found': False, 'disease': None, 'distance': 0.0,
            'threshold': 0.0, 'counts': {}, 'error': str(e),
        }

    best_cls  = None
    best_dist = float('inf')
    best_thr  = 0.0
    counts    = {}

    for cls_name in os.listdir(few_shot_dir):
        cls_path = os.path.join(few_shot_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        feats = _get_eff_class_features(cls_path, cls_name)
        if feats is None:
            continue
        counts[cls_name] = feats.shape[0]

        dist, thr = _mahal_dist_and_threshold(query_feat, feats)
        if dist < best_dist:
            best_dist = dist
            best_thr  = thr
            best_cls  = cls_name

    # Seuil infini = classe avec N < 2 → identification impossible
    thr_valid = (best_thr != float('inf') and best_thr > 0)
    found     = (best_cls is not None) and thr_valid and (best_dist <= best_thr)

    return {
        'found'    : found,
        'disease'  : best_cls if found else None,
        'distance' : round(best_dist, 4) if best_dist != float('inf') else None,
        'threshold': round(best_thr, 4)  if thr_valid else None,
        'counts'   : counts,
    }


# =============================================================================
#  API publique — Ajout à la base few-shot
# =============================================================================
def add_to_few_shot_base(image_path: str, disease_name: str, few_shot_dir: str) -> bool:
    """
    Copie une image dans le dossier few-shot de la classe donnée.
    Crée le dossier si nécessaire.

    Retourne True si succès, False en cas d'erreur.
    """
    dest = os.path.join(few_shot_dir, disease_name.replace(' ', '_'))
    os.makedirs(dest, exist_ok=True)
    try:
        shutil.copy2(image_path, os.path.join(dest, os.path.basename(image_path)))
        return True
    except Exception:
        return False


# =============================================================================
#  API publique — Ré-entraînement few-shot
# =============================================================================
def retrain(few_shot_per_class_dir: str) -> bool:
    """
    Ré-entraînement few-shot pour N classes dynamiques.
    Ne modifie PAS l'architecture EfficientNetB1 (extracteur figé).

    Met à jour en mémoire et sur disque :
      - _features_by_class : dict {classe: np.ndarray (N, 1280)}
      - _threshold_energy  : seuil Energy Distance (percentile 10 intra-classe)
      - _mahal_stats       : mean + inv_cov par classe + seuil global Mahalanobis

    Paramètres :
        few_shot_per_class_dir : dossier avec sous-dossiers par classe

    Retourne : True si succès
    """
    global _features_by_class, _threshold_energy, _mahal_stats, CLASS_NAMES

    _load_model()
    _load_energy_data()   # charge _features_by_class et _threshold_energy

    print("[retrain EfficientNet] Extraction des features few-shot…")

    # ── 1. Copier les features originales (base de référence) ─────────────────
    new_features = {k: np.array(v, dtype=np.float64) for k, v in _features_by_class.items()}

    # ── 2. Ajouter les nouvelles classes depuis few_shot_per_class/ ───────────
    if os.path.isdir(few_shot_per_class_dir):
        for cls_name in os.listdir(few_shot_per_class_dir):
            cls_path = os.path.join(few_shot_per_class_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue
            imgs = sorted(
                f for f in os.listdir(cls_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            )
            if not imgs:
                continue
            feats = []
            for fname in imgs:
                try:
                    feats.append(_extract_eff_feature(os.path.join(cls_path, fname)))
                except Exception as e:
                    print(f"  [skip] {fname}: {e}")
            if feats:
                new_features[cls_name] = np.array(feats, dtype=np.float64)   # (N, 1280)
                print(f"  classe '{cls_name}' : {len(feats)} images")

    # ── 3. Recalculer le seuil Energy (percentile 10 intra-classe) ────────────
    # Seuil plus conservateur (10e percentile) pour réduire les faux positifs OOD
    intra_ed = []
    for cls, feats in new_features.items():
        for i in range(len(feats)):
            others = np.delete(feats, i, axis=0)
            if len(others) > 0:
                intra_ed.append(_energy_distance(feats[i:i+1], others))
    new_energy_thr = float(np.percentile(intra_ed, 10)) if intra_ed else float(_threshold_energy)

    # ── 4. Recalculer les stats Mahalanobis pour toutes les classes ───────────
    new_mahal = {}
    reg       = 1e-4

    for cls_name, feats in new_features.items():
        n, d = feats.shape
        mean = feats.mean(axis=0)

        if n < 2:
            inv_cov = np.eye(d) / reg
        elif n < max(10, d // 4):
            var     = feats.var(axis=0) + reg
            inv_cov = np.diag(1.0 / var)
        else:
            cov = np.cov(feats, rowvar=False) + reg * np.eye(d)
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv_cov = np.diag(1.0 / (np.diag(cov) + reg))

        new_mahal[cls_name] = {'mean': mean, 'inv_cov': inv_cov}

    # Seuil Mahal global = percentile 90 des distances intra-classe
    intra_mahal = []
    for cls_name, stats in new_mahal.items():
        feats = new_features[cls_name]
        for f in feats:
            try:
                intra_mahal.append(float(sp_mahal(f, stats['mean'], stats['inv_cov'])))
            except Exception:
                continue
    new_mahal['_threshold'] = float(np.percentile(intra_mahal, 90)) if intra_mahal else 15.0

    # ── 5. Sauvegarder les fichiers PKL sur disque ────────────────────────────
    with open(FEATURES_PATH, 'wb') as f:
        pickle.dump(new_features, f)
    with open(THRESHOLD_ENERGY_PATH, 'wb') as f:
        pickle.dump(new_energy_thr, f)
    with open(MAHAL_STATS_PATH, 'wb') as f:
        pickle.dump(new_mahal, f)

    # ── 6. Mettre à jour les singletons en mémoire ────────────────────────────
    _features_by_class = new_features
    _threshold_energy  = new_energy_thr
    _mahal_stats       = new_mahal
    CLASS_NAMES        = list(new_features.keys())

    print(f"[retrain EfficientNet] {len(CLASS_NAMES)} classes : {CLASS_NAMES}")
    print(f"[retrain EfficientNet] Seuil Energy    : {new_energy_thr:.4f}")
    print(f"[retrain EfficientNet] Seuil Mahal     : {new_mahal['_threshold']:.4f}")
    return True
