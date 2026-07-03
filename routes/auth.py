# =============================================================================
#  routes/auth.py
#  Blueprint Flask pour l'authentification — connexion et déconnexion.
#  Gère la redirection automatique vers le tableau de bord approprié
#  selon le rôle de l'utilisateur (médecin ou chef de service).
# =============================================================================

# ─── Framework Flask ──────────────────────────────────────────────────────────
from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user

# ─── Modèles locaux ───────────────────────────────────────────────────────────
from models import User   # modèle utilisateur avec vérification du mot de passe

# =============================================================================
#  Blueprint
# =============================================================================
auth_bp = Blueprint('auth', __name__)


# =============================================================================
#  Route — Page de connexion
# =============================================================================
@auth_bp.route('/', methods=['GET', 'POST'])
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """
    Page de connexion de l'application PneumoIA.
    GET  → affiche le formulaire de connexion.
    POST → vérifie les identifiants et redirige selon le rôle.

    Règles :
    - Un utilisateur déjà connecté est redirigé directement vers son tableau de bord.
    - Seuls les comptes actifs (is_active_account=True) peuvent se connecter.
    - La recherche se fait par email (insensible à la casse).
    """
    # Redirection immédiate si l'utilisateur est déjà authentifié
    if current_user.is_authenticated:
        return _redirect_by_role(current_user)

    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        # Vérification : compte actif avec le bon email
        user = User.query.filter_by(email=email, is_active_account=True).first()

        if user and user.check_password(password):
            # Connexion réussie — session non persistante (remember=False)
            login_user(user, remember=False)
            return _redirect_by_role(user)

        # Email introuvable, compte désactivé ou mot de passe incorrect
        flash('Email ou mot de passe incorrect.', 'danger')

    return render_template('login.html')


# =============================================================================
#  Route — Déconnexion
# =============================================================================
@auth_bp.route('/logout')
@login_required
def logout():
    """
    Déconnecte l'utilisateur courant et redirige vers la page de connexion.
    Protégé par @login_required pour éviter les requêtes non authentifiées.
    """
    logout_user()
    flash('Vous avez été déconnecté.', 'info')
    return redirect(url_for('auth.login'))


# =============================================================================
#  Helper — Redirection selon le rôle
# =============================================================================
def _redirect_by_role(user):
    """
    Redirige vers le tableau de bord approprié selon le rôle de l'utilisateur.
    - Chef de service → /admin/dashboard
    - Médecin         → /doctor/dashboard
    """
    if user.is_chef:
        return redirect(url_for('admin.dashboard'))
    return redirect(url_for('doctor.dashboard'))
