# =============================================================================
#  config.py
#  Configuration centralisée de l'application Flask PneumoIA.
#  Tous les paramètres de l'application sont définis ici pour éviter les
#  valeurs "en dur" dans le reste du code.
# =============================================================================

# ─── Bibliothèques standard ───────────────────────────────────────────────────
import os        # manipulation des chemins de fichiers
import secrets   # génération de clés secrètes sécurisées

# Répertoire racine du projet (dossier contenant ce fichier config.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    """
    Classe de configuration principale.
    Les variables d'environnement ont priorité sur les valeurs par défaut.
    """

    # ── Sécurité ──────────────────────────────────────────────────────────────
    # Clé secrète pour signer les sessions Flask et les tokens CSRF.
    # En production, définir la variable d'environnement SECRET_KEY.
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(32)

    # ── Base de données ───────────────────────────────────────────────────────
    # SQLite local — suffisant pour un usage hospitalier mono-site.
    # Pour une mise en production multi-serveurs, utiliser PostgreSQL/MySQL.
    SQLALCHEMY_DATABASE_URI       = f"sqlite:///{os.path.join(BASE_DIR, 'medical_ai.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # désactiver les signaux de modification (performance)

    # ── Dossiers de stockage ──────────────────────────────────────────────────
    # Images radiologiques uploadées par les médecins
    UPLOAD_FOLDER             = os.path.join(BASE_DIR, 'uploads')

    # Images few-shot soumises pour le ré-entraînement (stockage plat)
    FEW_SHOT_FOLDER           = os.path.join(BASE_DIR, 'few_shot_data')

    # Cartes d'explicabilité (GradCAM++, SHAP, Hybrid, Segmentation) — servi via /static/
    EXPLAINABILITY_FOLDER     = os.path.join(BASE_DIR, 'static', 'explainability')
    EXPLAINABILITY_DIR        = os.path.join(BASE_DIR, 'static', 'explainability')

    # Images few-shot organisées par classe (sous-dossiers) — utilisé pour la comparaison few-shot
    FEW_SHOT_PER_CLASS_FOLDER = os.path.join(BASE_DIR, 'few_shot_per_class')

    # ── Limites upload ────────────────────────────────────────────────────────
    # Taille maximale d'un fichier uploadé (32 Mo — suffisant pour les DICOM)
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024   # 32 MB

    # Extensions acceptées pour les images radiologiques
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'dcm'}

    # ── Paramètres few-shot / ré-entraînement ─────────────────────────────────
    # Seuil global (legacy) — nombre total d'images avant ré-entraînement
    FEW_SHOT_RETRAINING_THRESHOLD = 1000

    # Seuil par classe — nombre d'images par maladie pour déclencher le ré-entraînement
    # automatique. Quand une classe atteint ce seuil, le ré-entraînement est lancé
    # automatiquement en arrière-plan (sans intervention du chef de service).
    FEW_SHOT_PER_CLASS_THRESHOLD  = 1000

    # Seuil de similarité cosinus pour la comparaison few-shot
    # (0.72 = 72 % de similarité minimale pour identifier une maladie)
    FEW_SHOT_SIM_THRESHOLD        = 0.72
