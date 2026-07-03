# =============================================================================
#  Définit les 4 tables de la base de données :
#    - User           : comptes médecins et chefs de service
#    - Prediction     : analyses radiologiques et leurs résultats
#    - FewShotSample  : images soumises pour le ré-entraînement few-shot
#    - RetrainingLog  : journal des opérations de ré-entraînement
# =============================================================================

# ─── Bibliothèques standard ───────────────────────────────────────────────────
from datetime import datetime

# ─── Flask-Login (mixin pour la gestion des sessions) ────────────────────────
from flask_login import UserMixin

# ─── Werkzeug (hachage sécurisé des mots de passe) ───────────────────────────
from werkzeug.security import generate_password_hash, check_password_hash

# ─── Extension locale (instance SQLAlchemy) ──────────────────────────────────
from extensions import db


# =============================================================================
#  Modèle — Utilisateur (médecin ou chef de service)
# =============================================================================
class User(UserMixin, db.Model):
    """
    Représente un utilisateur du système.
    Rôles possibles : 'medecin' | 'chef_service'

    UserMixin fournit les méthodes requises par Flask-Login :
    is_authenticated, is_active, is_anonymous, get_id()
    """
    __tablename__ = 'users'

    id                = db.Column(db.Integer, primary_key=True)
    username          = db.Column(db.String(80), unique=True, nullable=False)
    email             = db.Column(db.String(120), unique=True, nullable=False)
    password_hash     = db.Column(db.String(256), nullable=False)

    # Mot de passe en clair (affiché au chef de service pour la gestion des comptes)
    plain_password    = db.Column(db.String(200), nullable=False)

    # Rôle : 'medecin' ou 'chef_service'
    role              = db.Column(db.String(20), nullable=False)

    # Compte créé par (chef de service — clé étrangère auto-référentielle)
    created_by        = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at        = db.Column(db.DateTime, default=datetime.utcnow)

    # Soft-delete : False = compte désactivé (pas supprimé physiquement)
    is_active_account = db.Column(db.Boolean, default=True)

    # Relation avec les prédictions du médecin
    predictions = db.relationship(
        'Prediction', backref='doctor', lazy='dynamic',
        foreign_keys='Prediction.doctor_id'
    )

    def set_password(self, password: str):
        """Hache et stocke le mot de passe. Garde aussi la version en clair."""
        self.password_hash  = generate_password_hash(password)
        self.plain_password = password

    def check_password(self, password: str) -> bool:
        """Vérifie un mot de passe contre le hash stocké."""
        return check_password_hash(self.password_hash, password)

    @property
    def is_chef(self) -> bool:
        """Vrai si l'utilisateur est chef de service."""
        return self.role == 'chef_service'

    @property
    def is_medecin(self) -> bool:
        """Vrai si l'utilisateur est médecin."""
        return self.role == 'medecin'

    def __repr__(self):
        return f'<User {self.username} [{self.role}]>'


# =============================================================================
#  Modèle — Prédiction (analyse radiologique)
# =============================================================================
class Prediction(db.Model):
    """
    Enregistre le résultat complet d'une analyse radiologique par IA.
    Contient :
    - les informations de la prédiction (maladie, confiance, OOD)
    - les fichiers d'explicabilité (GradCAM++, SHAP, Hybrid, Segmentation)
    - les métriques de validation par perturbation
    - le diagnostic final validé par le médecin
    """
    __tablename__ = 'predictions'

    id             = db.Column(db.Integer, primary_key=True)
    doctor_id      = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_filename = db.Column(db.String(500), nullable=False)

    # Modèle IA utilisé : 'vit' ou 'efficient'
    model_used  = db.Column(db.String(20), nullable=False)

    # Méthode OOD pour EfficientNet : 'mahalanobis' | 'energy' (None pour ViT)
    method_used = db.Column(db.String(20), nullable=True)

    # ── Résultat de la prédiction ─────────────────────────────────────────────
    predicted_disease = db.Column(db.String(200), nullable=True)
    confidence        = db.Column(db.Float, nullable=True)   # valeur en % (0-100, PAS 0-1)

    # ── Détection hors-distribution (OOD) ────────────────────────────────────
    is_known      = db.Column(db.Boolean, nullable=False)    # False = image OOD
    ood_distance  = db.Column(db.Float, nullable=True)       # Energy Distance ou distance Mahalanobis
    ood_threshold = db.Column(db.Float, nullable=True)       # seuil utilisé pour le verdict OOD

    # ── Fichiers d'explicabilité (noms de fichiers dans static/explainability/) ─
    gradcam_filename      = db.Column(db.String(500), nullable=True)
    shap_filename         = db.Column(db.String(500), nullable=True)
    hybrid_filename       = db.Column(db.String(500), nullable=True)
    segmentation_filename = db.Column(db.String(500), nullable=True)

    # ── Métriques de validation par perturbation (chute de confiance en %) ───
    # Principe : masquer les zones actives → si la confiance chute fortement (>20%),
    # les zones identifiées sont bien responsables de la décision.
    perturb_drop_gradcam = db.Column(db.Float, nullable=True)
    perturb_drop_shap    = db.Column(db.Float, nullable=True)
    perturb_drop_hybrid  = db.Column(db.Float, nullable=True)

    # ── Scores qualitatifs des cartes d'activation ────────────────────────────
    sparsity_score = db.Column(db.Float, nullable=True)  # ratio de pixels actifs
    focus_score    = db.Column(db.Float, nullable=True)  # concentration de l'activation

    # ── Validation par le médecin ──────────────────────────────────────────────
    # validated : None = en attente | True = confirmé | False = corrigé
    validated     = db.Column(db.Boolean, nullable=True)
    final_disease = db.Column(db.String(200), nullable=True)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    # Relation avec le sample few-shot associé (si soumis)
    few_shot_sample = db.relationship('FewShotSample', backref='prediction', uselist=False)

    @property
    def status_label(self) -> str:
        """Libellé textuel du statut de validation."""
        if self.validated is None:
            return 'En attente'
        return 'Validé' if self.validated else 'Corrigé'

    @property
    def status_class(self) -> str:
        """
        Classe Bootstrap CSS pour la couleur du badge de statut.
        - En attente → 'danger' (rouge)
        - Validé     → 'success' (vert)
        - Corrigé    → 'danger' (rouge)
        """
        if self.validated is None:
            return 'danger'
        return 'success' if self.validated else 'danger'

    @property
    def ood_pct(self):
        """
        Pourcentage de la distance OOD par rapport au seuil.
        Utilisé pour l'affichage de la barre de progression OOD.
        Retourne None si les données sont manquantes.
        """
        if self.ood_distance is None or self.ood_threshold is None or self.ood_threshold == 0:
            return None
        return round(self.ood_distance / self.ood_threshold * 100, 1)

    def __repr__(self):
        return f'<Prediction {self.id} {self.model_used}>'


# =============================================================================
#  Modèle — Échantillon Few-Shot
# =============================================================================
class FewShotSample(db.Model):
    """
    Enregistre une image soumise par un médecin pour le ré-entraînement few-shot.
    C'est la SOURCE DE VÉRITÉ pour le comptage par classe (pas le système de fichiers).
    Chaque entrée correspond à une image OOD identifiée et nommée par le médecin.
    """
    __tablename__ = 'few_shot_samples'

    id             = db.Column(db.Integer, primary_key=True)
    prediction_id  = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=True)
    disease_name   = db.Column(db.String(200), nullable=False)   # nom de la maladie fourni par le médecin
    image_filename = db.Column(db.String(500), nullable=False)   # nom du fichier image
    method         = db.Column(db.String(20),  nullable=False)   # 'mahalanobis' | 'energy'
    added_at       = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<FewShot {self.disease_name} [{self.method}]>'


# =============================================================================
#  Modèle — Journal de ré-entraînement
# =============================================================================
class RetrainingLog(db.Model):
    """
    Journal des opérations de ré-entraînement des modèles IA.
    Statuts possibles : 'pending' → 'running' → 'completed' | 'failed'
    Utilisé dans le tableau de bord du chef de service pour suivre l'avancement.
    """
    __tablename__ = 'retraining_logs'

    id           = db.Column(db.Integer, primary_key=True)
    triggered_at = db.Column(db.DateTime, default=datetime.utcnow)   # date de déclenchement
    sample_count = db.Column(db.Integer)                              # nombre d'images utilisées
    status       = db.Column(db.String(20), default='pending')        # 'pending'|'running'|'completed'|'failed'
    triggered_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)              # date de fin (si terminé)
    notes        = db.Column(db.Text, nullable=True)                  # messages d'erreur ou informations

    def __repr__(self):
        return f'<Retraining {self.id} {self.status}>'
