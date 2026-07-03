# =============================================================================
#  run_flask.py
#  Point d'entrée de l'application PneumoIA en production.
#  Utilise Waitress (WSGI server) au lieu du serveur de développement Flask
#  pour la stabilité et les performances en environnement hospitalier.
#
#  Lancement :
#      python run_flask.py          (avec fenêtre console)
#      pythonw run_flask.py         (sans fenêtre console, processus persistant)
#
#  Pour arrêter :
#      Gestionnaire des tâches Windows → processus pythonw.exe → Fin de tâche
# =============================================================================

# ─── Application Flask ────────────────────────────────────────────────────────
from app import create_app          # factory function qui crée l'application

# ─── Serveur WSGI Waitress ────────────────────────────────────────────────────
from waitress import serve          # serveur WSGI stable pour Windows

# Créer l'instance de l'application Flask
app = create_app()

# Activer le rechargement automatique des templates Jinja2 lors des modifications
# (utile en développement, inoffensif en production)
app.jinja_env.auto_reload = True

# ── Démarrage du serveur ──────────────────────────────────────────────────────
print(" * PneumoIA — démarrage avec Waitress (WSGI)")
print(" * Accès local  : http://127.0.0.1:8080")
print(" * Accès réseau : http://0.0.0.0:8080")
print(" * Appuyez sur CTRL+C pour arrêter (si lancé en mode console)")

serve(
    app,
    host    = '0.0.0.0',   # écouter sur toutes les interfaces réseau
    port    = 8080,         # port d'écoute (différent du port Flask dev 5000)
    threads = 4,            # 4 threads pour gérer les requêtes simultanées
)
