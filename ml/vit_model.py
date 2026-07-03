"""
=============================================================================
 ml/vit_model.py
 Pipeline complet ViT (google/vit-base-patch16-224-in21k) pour PneumoIA.

 Étapes du pipeline :
   1. Prédiction  : ViTForImageClassification (logits → softmax)
   2. OOD         : Energy Distance inter-classes, seuil = percentile 95
   3. GradCAM++   : Similarité cosinus CLS-patch sur hidden states
   4. SHAP        : GradientExplainer + ViTShapWrapper (logits)
   5. Hybrid      : max(cam,shap) + 0.4*consensus + CLAHE
   6. Segmentation: morphologique + régions actives + overlay coloré
   7. Perturbation: masquage pixels > 0.5, intensity = 0.8
   8. Scores      : sparsity (ratio pixels actifs), focus (concentration)

 CONFIGURATION — 3 chemins à mettre à jour selon votre installation :
=============================================================================
"""

# ─── Bibliothèques standard Python ───────────────────────────────────────────
import os        # manipulation des chemins de fichiers
import uuid      # noms de fichiers uniques pour les cartes d'explicabilité
import pickle    # sérialisation/désérialisation des features et seuils
import warnings  # suppression des warnings SHAP non critiques

# ─── Calcul scientifique ──────────────────────────────────────────────────────
import numpy as np                                     # opérations matricielles
import cv2                                             # traitement d'images (resize, CLAHE, overlay)
from scipy import ndimage                              # opérations morphologiques (label, fill)
from scipy.spatial.distance import cdist              # distances entre ensembles de vecteurs
from scipy.spatial.distance import mahalanobis as sp_mahal  # distance de Mahalanobis

# ─── PyTorch (modèle ViT) ─────────────────────────────────────────────────────
import torch                                           # tenseurs et inférence
import torch.nn as nn                                  # modules de réseau de neurones
import torch.nn.functional as F                        # fonctions d'activation (softmax, cosine_similarity)

# ─── Vision — prétraitement des images ────────────────────────────────────────
from PIL import Image as PILImage                      # chargement des images radiologiques
from torchvision import transforms                     # transformations pour le background SHAP

# ─── Hugging Face Transformers (ViT) ─────────────────────────────────────────
from transformers import (
    ViTForImageClassification,   # modèle de classification fine-tuné
    ViTModel,                    # backbone ViT pour l'attention rollout
    ViTConfig,                   # configuration du modèle ViT
    AutoImageProcessor,          # préprocesseur d'images adapté au modèle
)

# ─── Explicabilité SHAP ───────────────────────────────────────────────────────
import shap   # SHapley Additive exPlanations (GradientExplainer)

# =============================================================================
#  Configuration — CHEMINS À METTRE À JOUR
# =============================================================================
MODEL_PATH     = r"C:\Users\hp\Desktop\newprjt\models\vit_model_4classes_best.pth"
FEATURES_PATH  = r"C:\Users\hp\Desktop\newprjt\models\vit_features_by_class.pkl"
THRESHOLD_PATH = r"C:\Users\hp\Desktop\newprjt\models\vit_threshold.pkl"
TRAIN_DIR      = r"C:\Users\hp\OneDrive\LungDisease_4Classes\train"   # images de fond pour SHAP
MODEL_NAME     = "google/vit-base-patch16-224-in21k"                  # identifiant Hugging Face

# =============================================================================
#  Constantes du modèle
# =============================================================================
ORIGINAL_CLASSES = ['Corona Virus Disease', 'Normal', 'Pneumonia', 'Tuberculosis']
CLASS_NAMES      = list(ORIGINAL_CLASSES)   # mis à jour dynamiquement après ré-entraînement
IMG_SIZE         = 224                       # taille des images en entrée du modèle
NORM_MEAN        = [0.5, 0.5, 0.5]          # normalisation ViT standard
NORM_STD         = [0.5, 0.5, 0.5]

# Répertoire de sortie pour les cartes d'explicabilité (défini par _inject_expl_dir)
EXPLAINABILITY_DIR = None

# =============================================================================
#  Singletons — chargés une seule fois (lazy loading)
# =============================================================================
_model              = None    # ViTForImageClassification (modèle de classification)
_backbone           = None    # ViTModel (backbone pour l'attention rollout)
_processor          = None    # AutoImageProcessor (préprocesseur d'images)
_shap_wrapper       = None    # _ViTShapWrapper (wrapper pour GradientExplainer)
_features_by_class  = None    # dict {classe: np.ndarray (N, 768)} — features de la base d'entraînement
_threshold          = None    # seuil Energy Distance pour la détection OOD
_device             = None    # torch.device (cuda ou cpu)

# Cache des features few-shot par classe — évite le recalcul à chaque comparaison
_few_shot_cache = {}   # {cls_name:count → np.ndarray (N, 768)}


# =============================================================================
#  Wrapper SHAP pour ViT
# =============================================================================
class _ViTShapWrapper(nn.Module):
    """
    Wrapper minimal autour de ViTForImageClassification pour SHAP.
    SHAP nécessite un module PyTorch dont forward() accepte directement
    le tensor d'image (pixel_values) et retourne les logits.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """Passe le tensor x comme pixel_values et retourne les logits bruts."""
        return self.model(pixel_values=x).logits


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
#  Chargement du modèle (lazy)
# =============================================================================
def _ready() -> bool:
    """Vérifie que tous les fichiers modèle nécessaires existent sur disque."""
    return all(os.path.exists(p) for p in [MODEL_PATH, FEATURES_PATH, THRESHOLD_PATH])


def _load_all() -> bool:
    """
    Charge le modèle ViT, le backbone, le processeur et les features de référence.
    Utilise un pattern singleton : ne charge qu'une seule fois par processus.
    Retourne False si les fichiers modèle sont introuvables (mode stub).
    """
    global _model, _backbone, _processor, _shap_wrapper
    global _features_by_class, _threshold, _device

    # Déjà chargé → retourner immédiatement
    if _model is not None:
        return True

    # Fichiers manquants → mode dégradé (stub)
    if not _ready():
        return False

    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)

    # Chargement des poids sauvegardés
    state = torch.load(MODEL_PATH, map_location=_device)

    # ── Modèle de classification ──────────────────────────────────────────────
    # Config locale pour éviter le téléchargement des poids Hugging Face
    config            = ViTConfig.from_pretrained(MODEL_NAME)
    config.num_labels = len(CLASS_NAMES)
    _model            = ViTForImageClassification(config)
    _model.load_state_dict(state, strict=False)
    _model.to(_device).eval()

    # ── Backbone pour l'attention (GradCAM via hidden states) ─────────────────
    bb_config                  = ViTConfig.from_pretrained(MODEL_NAME)
    bb_config.add_pooling_layer = False
    _backbone                  = ViTModel(bb_config)
    # Extraire uniquement les poids du backbone (préfixe "vit.")
    backbone_state = {k.replace("vit.", "", 1): v for k, v in state.items() if k.startswith("vit.")}
    _backbone.load_state_dict(backbone_state, strict=False)
    _backbone.to(_device).eval()

    # ── Wrapper SHAP ──────────────────────────────────────────────────────────
    _shap_wrapper = _ViTShapWrapper(_model).to(_device).eval()

    # ── Chargement des features de référence et du seuil OOD ─────────────────
    with open(FEATURES_PATH, "rb") as f:
        _features_by_class = pickle.load(f)
    with open(THRESHOLD_PATH, "rb") as f:
        _threshold = pickle.load(f)

    return True


# =============================================================================
#  Energy Distance (OOD)
# =============================================================================
def _energy_distance_point_vs_class(x: np.ndarray, class_feats: np.ndarray) -> float:
    """
    Calcule l'Energy Distance entre un point x et un ensemble de points Y.
    Formule : ED = 2·E[||x-Y||] - E[||Y-Y'||]  (d_xx = 0 car un seul point)

    Paramètres :
        x           : vecteur query (768,)
        class_feats : matrice de features de la classe (N, 768)
    """
    x    = x.reshape(1, -1)
    d_xy = cdist(x, class_feats, metric='euclidean').mean()
    d_yy = cdist(class_feats, class_feats, metric='euclidean').mean()
    return float(2 * d_xy - d_yy)


# =============================================================================
#  GradCAM++ via similarité cosinus CLS-patch
# =============================================================================
def _compute_gradcam(pil_img) -> np.ndarray:
    """
    GradCAM ViT via similarité cosinus entre le token CLS et les tokens patches.
    Utilise la moyenne des 4 dernières couches hidden states pour une
    heatmap plus riche et stable.

    Contourne le problème SDPA (transformers >= 4.36) qui ne retourne plus
    les poids d'attention traditionnels.

    Retourne : heatmap float32 (224, 224) dans [0, 1]
    """
    inputs = _processor(images=pil_img, return_tensors='pt')
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs, output_hidden_states=True)

    # Moyenne des 4 dernières couches pour une heatmap plus riche
    hidden_states = outputs.hidden_states      # tuple de tensors (1, seq_len, 768)
    layers        = hidden_states[-4:]         # 4 dernières couches
    stacked       = torch.stack([h[0] for h in layers], dim=0)   # (4, seq_len, 768)
    avg_hidden    = stacked.mean(dim=0)                           # (seq_len, 768)

    cls_token    = avg_hidden[0:1, :]    # token CLS (1, 768)
    patch_tokens = avg_hidden[1:,  :]   # tokens patches (num_patches, 768)

    # Similarité cosinus : CLS → chaque patch → heatmap 14x14
    sim = F.cosine_similarity(cls_token, patch_tokens, dim=-1)   # (num_patches,)
    cam = sim.cpu().numpy().astype(np.float32)

    side = int(np.sqrt(cam.shape[0]))    # 14 pour patch_size=16, img 224×224
    cam  = cam.reshape(side, side)
    cam  = np.maximum(cam, 0)
    cam  = cv2.resize(cam.astype(np.float32), (IMG_SIZE, IMG_SIZE))

    # Seuillage : garder uniquement les 35% pixels les plus activés
    cam = np.where(cam > np.percentile(cam, 65), cam, 0)
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # CLAHE pour accentuer le contraste local
    cam_u8 = np.uint8(255 * cam)
    clahe  = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    cam    = clahe.apply(cam_u8).astype(np.float32) / 255.0
    cam    = cv2.GaussianBlur(cam, (5, 5), 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype(np.float32)


# =============================================================================
#  SHAP — Background et calcul
# =============================================================================
def _get_shap_background(n: int = 5):
    """
    Charge quelques images du dataset d'entraînement comme background SHAP.
    Si le dossier train est introuvable, retourne un tensor de zéros.

    Paramètres :
        n : nombre d'images de fond à charger

    Retourne : tensor PyTorch (n, 3, 224, 224) sur _device
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    tensors = []
    if os.path.exists(TRAIN_DIR):
        for cls in CLASS_NAMES:
            cls_path = os.path.join(TRAIN_DIR, cls)
            if not os.path.exists(cls_path):
                continue
            for fname in os.listdir(cls_path)[:3]:
                try:
                    img = PILImage.open(os.path.join(cls_path, fname)).convert('RGB')
                    tensors.append(transform(img))
                    if len(tensors) >= n:
                        break
                except Exception:
                    pass
            if len(tensors) >= n:
                break

    if not tensors:
        # Fallback : tensor de zéros (background neutre)
        return torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(_device)
    return torch.stack(tensors[:n]).to(_device)


def _compute_shap(pil_img, background) -> np.ndarray:
    """
    Calcule la carte SHAP via GradientExplainer appliqué au ViTShapWrapper.
    Les valeurs absolues SHAP sont moyennées sur les dimensions et normalisées.

    Retourne : heatmap float32 (224, 224) dans [0, 1]
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    x_input = transform(pil_img).unsqueeze(0).to(_device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # supprimer les warnings SHAP non critiques
        explainer   = shap.GradientExplainer(_shap_wrapper, background)
        shap_values = explainer.shap_values(x_input)

    # Moyenne des valeurs absolues sur toutes les dimensions sauf H×W
    sv = np.array(shap_values)
    sv = np.abs(sv).mean(axis=0)
    while sv.ndim > 2:
        sv = sv.mean(axis=0)

    sv = np.nan_to_num(sv).astype(np.float32)
    sv = cv2.resize(sv, (IMG_SIZE, IMG_SIZE))

    # Seuillage des 35% pixels les moins activés
    sv = np.where(sv > np.percentile(sv, 65), sv, 0)
    if sv.max() > sv.min():
        sv = (sv - sv.min()) / (sv.max() - sv.min())

    # CLAHE pour améliorer le contraste
    sv_u8  = np.uint8(255 * sv)
    clahe  = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    sv     = clahe.apply(sv_u8).astype(np.float32) / 255.0
    sv     = cv2.GaussianBlur(sv, (5, 5), 0)
    if sv.max() > 0:
        sv = sv / sv.max()
    return sv.astype(np.float32)


# =============================================================================
#  Fusion Hybrid
# =============================================================================
def _fuse_maps(cam_map: np.ndarray, shap_map: np.ndarray) -> np.ndarray:
    """
    Fusion hybride des cartes GradCAM++ et SHAP :
        hybrid = max(cam, shap) + 0.4 × consensus
    Puis seuillage, CLAHE et lissage gaussien.

    La composante consensus (cam × shap) renforce les zones
    confirmées par les deux méthodes simultanément.

    Retourne : heatmap float32 (224, 224) dans [0, 1]
    """
    cam_map  = cv2.resize(cam_map.astype(np.float32),  (IMG_SIZE, IMG_SIZE))
    shap_map = cv2.resize(shap_map.astype(np.float32), (IMG_SIZE, IMG_SIZE))

    # Normalisation indépendante des deux maps
    if cam_map.max()  > 0: cam_map  = cam_map  / cam_map.max()
    if shap_map.max() > 0: shap_map = shap_map / shap_map.max()

    # Fusion : maximum des deux + composante consensus
    fused     = np.maximum(cam_map, shap_map)
    consensus = cam_map * shap_map
    if consensus.max() > 0:
        consensus = consensus / consensus.max()
    fused = fused + 0.4 * consensus

    # Seuillage des 55% pixels les moins activés
    fused = np.where(fused > np.percentile(fused, 45), fused, 0)
    if fused.max() > fused.min():
        fused = (fused - fused.min()) / (fused.max() - fused.min())

    # CLAHE pour améliorer le contraste
    fused_u8 = np.uint8(255 * fused)
    clahe    = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    fused    = clahe.apply(fused_u8).astype(np.float32) / 255.0
    fused    = cv2.GaussianBlur(fused, (3, 3), 0)
    if fused.max() > 0:
        fused = fused / fused.max()
    return fused.astype(np.float32)


# =============================================================================
#  Segmentation morphologique
# =============================================================================
def _segment(heatmap: np.ndarray, img_np: np.ndarray = None,
             threshold: float = 0.5, min_area: int = 200) -> np.ndarray:
    """
    Segmentation des zones actives par seuillage + opérations morphologiques.
    Les régions de moins de min_area pixels sont filtrées.

    Paramètres :
        heatmap   : carte d'activation float32 (H, W) dans [0, 1]
        img_np    : image originale float32 (H, W, 3) dans [0, 1] — pour l'overlay
        threshold : seuil de binarisation (défaut : 0.5)
        min_area  : surface minimale d'une région en pixels (défaut : 200)

    Retourne :
        - Si img_np fourni : image BGR avec overlay HOT et contour cyan
        - Sinon : masque vert sur fond noir
    """
    # Binarisation + opérations morphologiques (fermeture puis ouverture)
    binary = (heatmap >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)   # boucher les trous
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)   # supprimer le bruit

    # Filtrer les petites régions (artefacts non significatifs)
    labeled, n_regions = ndimage.label(binary)
    mask_filtered = np.zeros_like(binary)
    for i in range(1, n_regions + 1):
        region = (labeled == i).astype(np.uint8)
        if region.sum() >= min_area:
            mask_filtered |= region

    if img_np is not None:
        # ── Overlay coloré sur l'image originale ──────────────────────────────
        orig    = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        result  = (orig * 0.35).astype(np.uint8)           # fond assombri à 35%
        heat    = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_HOT)
        blended = cv2.addWeighted(orig, 0.35, heat, 0.65, 0)
        mask_bool = mask_filtered.astype(bool)
        result[mask_bool] = blended[mask_bool]              # zones actives en HOT

        # Contour cyan sur les régions segmentées
        contours, _ = cv2.findContours(mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 255), 2)
        return result

    # ── Fallback : vert sur fond noir ─────────────────────────────────────────
    result = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    result[mask_filtered.astype(bool)] = [0, 200, 100]
    contours, _ = cv2.findContours(mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 255), 2)
    return result


# =============================================================================
#  Validation par perturbation
# =============================================================================
def _perturb_drop(img_tensor, heatmap: np.ndarray, pred_idx: int,
                  orig_conf: float, intensity: float = 0.8) -> float:
    """
    Mesure la chute de confiance après masquage des zones actives.
    Principe : si le masquage des zones identifiées fait chuter la confiance
    de façon importante (> 20%), ces zones sont bien responsables de la décision.

    Paramètres :
        img_tensor : tensor PyTorch original (1, 3, H, W)
        heatmap    : carte d'activation float32 (H, W) dans [0, 1]
        pred_idx   : indice de la classe prédite
        orig_conf  : confiance originale en % (0-100)
        intensity  : intensité du masquage (0.8 = 80% d'obscurcissement)

    Retourne : chute de confiance en % (valeur positive = baisse)
    """
    h = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    h = h / (h.max() + 1e-8)

    # Masquer les pixels avec activation > 0.5
    mask      = torch.tensor(h > 0.5, dtype=torch.float32).to(_device)
    perturbed = img_tensor.clone()
    perturbed[0] = perturbed[0] * (1 - intensity * mask)

    with torch.no_grad():
        logits = _model(pixel_values=perturbed).logits

    p_conf = torch.softmax(logits, dim=1)[0, pred_idx].item() * 100
    return round(orig_conf - p_conf, 2)


# =============================================================================
#  Scores qualitatifs
# =============================================================================
def _sparsity_score(m: np.ndarray) -> float:
    """Ratio de pixels actifs dans la heatmap — plus c'est faible, plus l'attention est focalisée."""
    return float(np.sum(m > 0) / m.size)

def _focus_score(m: np.ndarray) -> float:
    """Rapport max/moyenne — mesure la concentration du pic d'activation."""
    return float(np.max(m) / (np.mean(m) + 1e-8))


# =============================================================================
#  Sauvegarde des cartes (utilitaires)
# =============================================================================
def _save(arr_bgr: np.ndarray, prefix: str) -> str:
    """Sauvegarde une image BGR dans EXPLAINABILITY_DIR avec un nom unique."""
    fname = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    cv2.imwrite(os.path.join(EXPLAINABILITY_DIR, fname), arr_bgr)
    return fname

def _heatmap_bgr(heatmap: np.ndarray, cmap=cv2.COLORMAP_JET) -> np.ndarray:
    """Convertit une heatmap float [0,1] en image BGR colorée."""
    return cv2.applyColorMap(np.uint8(255 * heatmap), cmap)

def _overlay_bgr(img_np_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Superpose la heatmap (JET) sur l'image originale avec alpha=0.5."""
    orig = cv2.cvtColor((img_np_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    heat = _heatmap_bgr(heatmap)
    return cv2.addWeighted(orig, 0.5, heat, 0.5, 0)


# =============================================================================
#  API publique — Prédiction rapide (sans explicabilité)
# =============================================================================
def predict(image_path: str) -> dict:
    """
    Prédiction de maladie + détection OOD uniquement.
    N'inclut PAS l'explicabilité (calculée séparément à la demande).

    Retourne un dict avec les clés :
        disease       : nom de la maladie prédite (None si OOD)
        confidence    : confiance en % (None si OOD)
        is_known      : True si in-distribution, False si OOD
        ood_distance  : Energy Distance calculée
        ood_threshold : seuil utilisé pour la décision OOD
    """
    if not _load_all():
        # ── Mode stub (fichiers modèle manquants) ─────────────────────────────
        import random
        print("[ViT] STUB — fichier modèle non trouvé.")
        is_k = random.random() > 0.25
        d    = round(random.uniform(3.0, 18.0), 4)
        thr  = 10.5
        return {
            'disease'      : random.choice(CLASS_NAMES) if is_k else None,
            'confidence'   : round(random.uniform(72, 99), 2) if is_k else None,
            'is_known'     : d <= thr,
            'ood_distance' : d,
            'ood_threshold': thr,
        }

    pil_img = PILImage.open(image_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))

    # ── Inférence ─────────────────────────────────────────────────────────────
    inputs = _processor(images=pil_img, return_tensors='pt')
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        out   = _model(**inputs, output_hidden_states=True)
        probs = torch.softmax(out.logits, dim=1)[0]

    # Features CLS pour le calcul OOD
    cls_feat = out.hidden_states[-1][:, 0, :].cpu().numpy()[0]   # (768,)

    # ── Energy Distance vers toutes les classes (originales + few-shot) ───────
    all_cls   = list(_features_by_class.keys())
    distances = {
        cls: _energy_distance_point_vs_class(cls_feat, _features_by_class[cls])
        for cls in all_cls
    }
    best_cls = min(distances, key=distances.get)
    best_ed  = distances[best_cls]
    is_known = best_ed <= float(_threshold)

    # ── Calcul de la confiance ─────────────────────────────────────────────────
    # Classe originale → score softmax du modèle (plus fiable)
    # Nouvelle classe few-shot → score de proximité Energy normalisé
    if is_known:
        if best_cls in ORIGINAL_CLASSES:
            orig_idx   = ORIGINAL_CLASSES.index(best_cls)
            confidence = float(probs[orig_idx].item() * 100) if orig_idx < len(probs) else None
        else:
            thr        = float(_threshold)
            confidence = round(max(0.0, min(100.0, (1.0 - best_ed / thr) * 100)), 2) if thr > 0 else None
    else:
        confidence = None   # OOD → pas de confiance fiable

    return {
        'disease'      : best_cls if is_known else None,
        'confidence'   : confidence,
        'is_known'     : is_known,
        'ood_distance' : round(best_ed, 6),
        'ood_threshold': round(float(_threshold), 6),
    }


# =============================================================================
#  API publique — Explicabilité à la demande
# =============================================================================
def compute_explainability(image_path: str, pred_idx: int = None) -> dict:
    """
    Calcule les cartes d'explicabilité GradCAM++, SHAP, Hybrid, Segmentation
    et les métriques de perturbation.
    Appelé séparément de predict(), à la demande du médecin ou du chef de service.

    Paramètres :
        image_path : chemin absolu vers l'image radiologique
        pred_idx   : indice de la classe à expliquer (None = classe la plus probable)

    Retourne un dict avec les clés :
        gradcam_filename, shap_filename, hybrid_filename, segmentation_filename,
        perturb_drop_gradcam, perturb_drop_shap, perturb_drop_hybrid
    """
    if not _load_all() or EXPLAINABILITY_DIR is None:
        return {}

    pil_img      = PILImage.open(image_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_np       = np.array(pil_img).astype(np.float32) / 255.0
    inputs       = _processor(images=pil_img, return_tensors='pt')
    inputs       = {k: v.to(_device) for k, v in inputs.items()}
    input_tensor = inputs['pixel_values']

    with torch.no_grad():
        out      = _model(**inputs, output_hidden_states=True)
        probs    = torch.softmax(out.logits, dim=1)[0]
        if pred_idx is None:
            pred_idx = int(torch.argmax(probs).item())

    confidence = float(probs[pred_idx].item() * 100)

    # ── GradCAM++ ─────────────────────────────────────────────────────────────
    gradcam_map = _compute_gradcam(pil_img)
    gc_fname    = _save(_overlay_bgr(img_np, gradcam_map), "gradcam_vit")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    try:
        background = _get_shap_background(n=5)
        shap_map   = _compute_shap(pil_img, background)
        sh_fname   = _save(_overlay_bgr(img_np, shap_map), "shap_vit")
    except Exception as e:
        print(f"[ViT SHAP] Erreur : {e} — fallback sur GradCAM++")
        shap_map = gradcam_map.copy()
        sh_fname = None

    # ── Hybrid ────────────────────────────────────────────────────────────────
    hybrid_map = _fuse_maps(gradcam_map, shap_map)
    hyb_fname  = _save(_overlay_bgr(img_np, hybrid_map), "hybrid_vit")

    # ── Segmentation ──────────────────────────────────────────────────────────
    seg_bgr   = _segment(hybrid_map, img_np=img_np, threshold=0.5)
    seg_fname = _save(seg_bgr, "seg_vit")

    # ── Validation par perturbation ───────────────────────────────────────────
    drop_gc  = _perturb_drop(input_tensor, gradcam_map, pred_idx, confidence)
    drop_sh  = _perturb_drop(input_tensor, shap_map,    pred_idx, confidence)
    drop_hyb = _perturb_drop(input_tensor, hybrid_map,  pred_idx, confidence)

    return {
        'gradcam_filename'      : gc_fname,
        'shap_filename'         : sh_fname,
        'hybrid_filename'       : hyb_fname,
        'segmentation_filename' : seg_fname,
        'perturb_drop_gradcam'  : drop_gc,
        'perturb_drop_shap'     : drop_sh,
        'perturb_drop_hybrid'   : drop_hyb,
    }


# =============================================================================
#  API publique — Ré-entraînement few-shot
# =============================================================================
def retrain(few_shot_per_class_dir: str) -> bool:
    """
    Ré-entraînement few-shot pour N classes dynamiques.
    Ne modifie PAS le modèle ViT (extracteur CLS figé à 4 classes originales).

    Met à jour en mémoire et sur disque :
      - _features_by_class : dict {classe: np.ndarray (N, 768)}
      - _threshold          : seuil Energy Distance (percentile 95 intra-classe)

    Paramètres :
        few_shot_per_class_dir : dossier avec sous-dossiers par classe

    Retourne : True si succès
    """
    global _features_by_class, _threshold, CLASS_NAMES

    if not _load_all():
        raise RuntimeError("ViT model non chargé — vérifiez les fichiers .pth et .pkl")

    print("[retrain ViT] Extraction des features few-shot…")

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
                    feats.append(_extract_cls_feature(os.path.join(cls_path, fname)))
                except Exception as e:
                    print(f"  [skip] {fname}: {e}")
            if feats:
                new_features[cls_name] = np.array(feats, dtype=np.float64)   # (N, 768)
                print(f"  classe '{cls_name}' : {len(feats)} images")

    # ── 3. Recalculer le seuil Energy (percentile 95 intra-classe) ────────────
    intra_ed = []
    for cls, feats in new_features.items():
        x = np.array(feats, dtype=np.float64)
        for i in range(len(x)):
            others = np.delete(x, i, axis=0)
            if len(others) > 0:
                intra_ed.append(_energy_distance_point_vs_class(x[i], others))
    new_threshold = float(np.percentile(intra_ed, 95)) if intra_ed else float(_threshold)

    # ── 4. Sauvegarder les fichiers PKL sur disque ────────────────────────────
    with open(FEATURES_PATH, 'wb') as f:
        pickle.dump(new_features, f)
    with open(THRESHOLD_PATH, 'wb') as f:
        pickle.dump(new_threshold, f)

    # ── 5. Mettre à jour les singletons en mémoire ────────────────────────────
    _features_by_class = new_features
    _threshold         = new_threshold
    CLASS_NAMES        = list(new_features.keys())

    print(f"[retrain ViT] {len(CLASS_NAMES)} classes : {CLASS_NAMES}")
    print(f"[retrain ViT] Seuil Energy Distance : {new_threshold:.4f}")
    return True


# =============================================================================
#  Helpers few-shot — Extraction de features
# =============================================================================
def _extract_cls_feature(image_path: str) -> np.ndarray:
    """
    Extrait le token CLS ViT d'une image.
    Retourne un vecteur numpy (768,) en float64.
    Utilisé pour le ré-entraînement few-shot et la comparaison.
    """
    pil = PILImage.open(image_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    inp = _processor(images=pil, return_tensors='pt')
    inp = {k: v.to(_device) for k, v in inp.items()}
    with torch.no_grad():
        out = _model(**inp, output_hidden_states=True)
    return out.hidden_states[-1][:, 0, :].squeeze(0).cpu().numpy().astype(np.float64)


def _get_class_features_np(cls_path: str, cls_name: str) -> np.ndarray | None:
    """
    Retourne la matrice de features (N, 768) pour une classe few-shot.
    Utilise un cache en mémoire indexé par (nom_classe, nombre_images)
    pour éviter le recalcul à chaque comparaison.
    """
    imgs = sorted(
        f for f in os.listdir(cls_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if not imgs:
        return None

    cache_key = f"{cls_name}:{len(imgs)}"
    if cache_key in _few_shot_cache:
        return _few_shot_cache[cache_key]

    feats = []
    for fname in imgs:
        try:
            feats.append(_extract_cls_feature(os.path.join(cls_path, fname)))
        except Exception:
            continue

    if not feats:
        return None

    arr                       = np.array(feats, dtype=np.float64)   # (N, 768)
    _few_shot_cache[cache_key] = arr
    return arr


def _mahal_dist_and_threshold(query: np.ndarray, feats: np.ndarray,
                               percentile: int = 95, reg: float = 1e-4):
    """
    Calcule la distance de Mahalanobis entre query et la distribution de la classe,
    ainsi que le seuil intra-classe au percentile donné.

    Stratégie adaptative selon le nombre d'échantillons :
        N < 2         : distance euclidienne, seuil = inf
        N < max(10, d/4) : covariance diagonale (plus stable en régime few-shot)
        Sinon         : covariance complète régularisée

    Retourne : (distance, threshold_intra_classe)
    """
    n, d   = feats.shape
    mean   = feats.mean(axis=0)

    if n < 2:
        # Pas assez d'exemples pour la covariance → distance euclidienne
        dist = float(np.linalg.norm(query - mean))
        return dist, float('inf')

    if n < max(10, d // 4):
        # Régime few-shot : covariance diagonale pour éviter la singularité
        var     = feats.var(axis=0) + reg
        inv_cov = np.diag(1.0 / var)
    else:
        # Régime normal : covariance complète régularisée
        cov = np.cov(feats, rowvar=False) + reg * np.eye(d)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Matrice singulière → fallback diagonal
            var     = np.diag(cov) + reg
            inv_cov = np.diag(1.0 / var)

    dist = float(sp_mahal(query, mean, inv_cov))

    # Seuil adaptatif selon le nombre d'échantillons disponibles :
    #   N ≤ 5  : très peu d'images → seuil = moyenne + 3×écart-type (règle des 3-sigma)
    #            évite que le percentile 95 sur 4-5 valeurs soit trop serré
    #   N < 20 : peu d'images → percentile 99 (plus permissif que 95)
    #   N ≥ 20 : assez d'images → percentile 95 (comportement standard)
    intra = [float(sp_mahal(f, mean, inv_cov)) for f in feats]
    arr   = np.array(intra)

    if n <= 5:
        std       = float(arr.std()) if arr.std() > 0 else float(arr.mean() * 0.5)
        threshold = float(arr.mean() + 3.0 * std)
    elif n < 20:
        threshold = float(np.percentile(arr, 99))
    else:
        threshold = float(np.percentile(arr, percentile))

    return dist, threshold


# =============================================================================
#  API publique — Comparaison few-shot
# =============================================================================
def few_shot_compare(image_path: str, few_shot_dir: str, **kwargs) -> dict:
    """
    Compare l'image avec la base few-shot via distance de Mahalanobis
    sur les features CLS ViT (768 dimensions).

    Retourne un dict avec les clés :
        found     : bool — True si une classe proche a été identifiée
        disease   : str|None — nom de la classe identifiée
        distance  : float — distance de Mahalanobis vers la meilleure classe
        threshold : float — seuil utilisé pour la décision
        counts    : dict {classe: nb_images} — contenu de la base few-shot
    """
    if not _load_all():
        return {'found': False, 'disease': None, 'distance': 0.0, 'threshold': 0.0, 'counts': {}}

    if not os.path.exists(few_shot_dir):
        return {'found': False, 'disease': None, 'distance': 0.0, 'threshold': 0.0, 'counts': {}}

    try:
        query_feat = _extract_cls_feature(image_path)   # (768,) float64
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
        feats = _get_class_features_np(cls_path, cls_name)
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
