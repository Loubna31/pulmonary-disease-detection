
import os
from flask import Flask, send_from_directory
from config import Config
from extensions import db, login_manager


# =============================================================================
#  Factory — Création de l'application
# =============================================================================
def create_app(config_class=Config):
    """
    Crée et configure une instance Flask.
    

    Étapes :
    1. Créer l'instance Flask et charger la configuration
    2. Créer les dossiers nécessaires (uploads, explainability, etc.)
    3. Initialiser les extensions (SQLAlchemy, LoginManager)
    4. Enregistrer les blueprints (auth, admin, doctor)
    5. Créer les tables DB et initialiser le compte admin par défaut

    Note sur les imports internes :
        Les blueprints et modèles sont importés DANS cette fonction
        (et non au niveau du module) pour respecter le pattern Flask Application
        Factory et éviter les imports circulaires potentiels lors des tests.
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    # ── Création des dossiers requis ──────────────────────────────────────────
    # Ces dossiers doivent exister avant le premier upload
    for folder in [
        app.config['UPLOAD_FOLDER'],
        app.config['FEW_SHOT_FOLDER'],
        app.config['EXPLAINABILITY_FOLDER'],
        app.config['FEW_SHOT_PER_CLASS_FOLDER'],
    ]:
        os.makedirs(folder, exist_ok=True)

    # ── Initialisation des extensions Flask ───────────────────────────────────
    db.init_app(app)
    login_manager.init_app(app)

    # ── User Loader (requis par Flask-Login) ──────────────────────────────────
    # Import interne : évite les problèmes d'import circulaire dans le contexte
    # de l'Application Factory
    from models import User

    @login_manager.user_loader
    def load_user(user_id):
        """Charge un utilisateur depuis la DB à partir de son ID de session."""
        return User.query.get(int(user_id))

    # ── Enregistrement des blueprints ─────────────────────────────────────────
    
    from routes.auth   import auth_bp
    from routes.admin  import admin_bp
    from routes.doctor import doctor_bp

    app.register_blueprint(auth_bp)                           # / et /login, /logout
    app.register_blueprint(admin_bp,  url_prefix='/admin')    # /admin/*
    app.register_blueprint(doctor_bp, url_prefix='/doctor')   # /doctor/*

    # ── Route pour servir les images uploadées ────────────────────────────────
    # Les images sont stockées hors du dossier /static pour des raisons de sécurité
    # et de séparation des préoccupations. Cette route les expose via /uploads/
    @app.route('/uploads/<path:filename>')
    def uploaded_file(filename):
        """Sert les images radiologiques uploadées depuis le dossier uploads/."""
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    # ── Initialisation de la base de données ──────────────────────────────────
    with app.app_context():
        db.create_all()          # crée les tables si elles n'existent pas
        _seed_default_admin()    # crée le compte admin par défaut si absent

    return app


# =============================================================================
#  Seeding — Compte administrateur par défaut
# =============================================================================
def _seed_default_admin():
    """
    Crée un compte chef de service par défaut si aucun n'existe.
    Appelé au démarrage de l'application pour garantir qu'il y a toujours
    au moins un compte administrateur opérationnel.

    Identifiants par défaut :
        Email    : admin@hopital.dz
        Password : Admin@123
    """
    
    from models import User

    if not User.query.filter_by(role='chef_service').first():
        chef = User(
            username = 'admin',
            email    = 'admin@hopital.dz',
            role     = 'chef_service',
        )
        chef.set_password('Admin@123')
        db.session.add(chef)
        db.session.commit()
        print('✓ Compte chef de service par défaut créé : admin@hopital.dz / Admin@123')
