# =============================================================================
#  extensions.py
#  Initialisation des extensions Flask partagées entre tous les modules.
#  Ces objets sont créés ici (sans app) puis attachés à l'application
#  dans app.py via la méthode init_app() (pattern "Application Factory").
# =============================================================================

# ─── Extensions Flask ─────────────────────────────────────────────────────────
from flask_sqlalchemy import SQLAlchemy    # ORM base de données
from flask_login import LoginManager       # gestion des sessions utilisateur

# ── Base de données SQLAlchemy ────────────────────────────────────────────────
db = SQLAlchemy()

# ── Gestionnaire de sessions (authentification) ───────────────────────────────
login_manager = LoginManager()

# Route de redirection quand un utilisateur non connecté tente d'accéder
# à une page protégée par @login_required
login_manager.login_view = 'auth.login'

# Message flash affiché lors de la redirection vers la page de connexion
login_manager.login_message          = 'Veuillez vous connecter pour accéder à cette page.'
login_manager.login_message_category = 'warning'
