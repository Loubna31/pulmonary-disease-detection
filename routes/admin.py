


import os           # manipulation des chemins (UPLOAD_FOLDER, EXPLAINABILITY_DIR)
import io           # flux mémoire pour la génération CSV en mémoire
import csv          # écriture/lecture de fichiers CSV
import datetime     # horodatage des exports et logs
import threading    # ré-entraînement asynchrone en arrière-plan
from functools import wraps  # décorateur @chef_required

# ─── SQLAlchemy (agrégation / GROUP BY) ───────────────────────────────────────
from sqlalchemy import func

# ─── Framework Flask ──────────────────────────────────────────────────────────
from flask import (
    Blueprint,
    render_template,
    redirect,
    url_for,
    flash,
    request,
    current_app,
    jsonify,
    Response,
)
from flask_login import login_required, current_user

# ─── Extensions et modèles locaux ─────────────────────────────────────────────
from extensions import db
from models import User, Prediction, FewShotSample, RetrainingLog

# ─── Helpers ML partagés avec le blueprint médecin ────────────────────────────

from routes.doctor import (
    _inject_expl_dir,           # injecte le répertoire d'explicabilité dans les modules ML
    _run_explainability_only,   # calcule GradCAM++/SHAP à la demande
    _generate_interpretation,   # génère le texte d'interprétation clinique
)

# ─── Modules ML pour le ré-entraînement ───────────────────────────────────────
from ml.efficient_model import retrain as eff_retrain   # ré-entraînement EfficientNet
from ml.vit_model       import retrain as vit_retrain   # ré-entraînement ViT

# ─── Génération de PDF ────────────────────────────────────────────────────────
try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    FPDF_AVAILABLE = True
except ImportError:
    # fpdf2 n'est pas installé — l'export PDF sera désactivé avec un message d'erreur
    FPDF_AVAILABLE = False

# =============================================================================
#  Blueprint
# =============================================================================
admin_bp = Blueprint('admin', __name__)


# =============================================================================
#  Décorateur d'accès — chef de service uniquement
# =============================================================================
def chef_required(f):
    """
    Décorateur qui restreint l'accès aux utilisateurs ayant le rôle 'chef_service'.
    Redirige vers le tableau de bord médecin si l'utilisateur est un médecin.
    """
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if not current_user.is_chef:
            flash('Accès réservé au chef de service.', 'danger')
            return redirect(url_for('doctor.dashboard'))
        return f(*args, **kwargs)
    return decorated


# =============================================================================
#  Routes — Tableau de bord chef de service
# =============================================================================
@admin_bp.route('/dashboard')
@chef_required
def dashboard():
    """
    Page principale du chef de service.
    Affiche :
    - les statistiques globales (médecins actifs, analyses, validées, inconnues, corrigées)
    - la progression few-shot par classe 
    - les informations du dernier ré-entraînement
    - la répartition des maladies détectées
    - les analyses récentes 
    """
    # ── Statistiques globales ─────────────────────────────────────────────────
    total_medecins    = User.query.filter_by(role='medecin', is_active_account=True).count()
    total_predictions = Prediction.query.count()
    total_validated   = Prediction.query.filter_by(validated=True).count()
    total_unknown     = Prediction.query.filter_by(is_known=False).count()

    # Corrigées = médecin a fourni un diagnostic alternatif (validated=False + final_disease renseigné)
    total_corrected = (
        Prediction.query
        .filter_by(validated=False, is_known=True)
        .filter(Prediction.final_disease.isnot(None))
        .count()
    )

    # Dernier ré-entraînement (pour afficher le statut dans le dashboard)
    last_retrain     = RetrainingLog.query.order_by(RetrainingLog.triggered_at.desc()).first()
    per_class_thresh = current_app.config.get('FEW_SHOT_PER_CLASS_THRESHOLD', 400)

    # ── Comptage few-shot par classe (source de vérité : table FewShotSample) ─
    rows_fs = (
        db.session.query(FewShotSample.disease_name, func.count(FewShotSample.id))
        .group_by(FewShotSample.disease_name)
        .order_by(func.count(FewShotSample.id).desc())
        .all()
    )
    per_class_counts = {disease: count for disease, count in rows_fs}

    # ── Répartition des maladies prédites (cas connus uniquement) ─────────────
    rows = (
        db.session.query(Prediction.predicted_disease, func.count(Prediction.id))
        .filter(
            Prediction.is_known == True,
            Prediction.predicted_disease.isnot(None),
        )
        .group_by(Prediction.predicted_disease)
        .order_by(func.count(Prediction.id).desc())
        .all()
    )
    disease_counts = {disease: count for disease, count in rows}

    # ── Analyses récentes (toutes, sans pagination) ───────────────────────────
    recent_predictions = (
        Prediction.query
        .order_by(Prediction.created_at.desc())
        .all()
    )

    return render_template(
        'admin/dashboard.html',
        total_medecins=total_medecins,
        total_predictions=total_predictions,
        total_validated=total_validated,
        total_unknown=total_unknown,
        total_corrected=total_corrected,
        last_retrain=last_retrain,
        recent_predictions=recent_predictions,
        per_class_counts=per_class_counts,
        per_class_thresh=per_class_thresh,
        disease_counts=disease_counts,
    )


# =============================================================================
#  Routes — Résultat d'une analyse (vue chef de service)
# =============================================================================
@admin_bp.route('/result/<int:pred_id>')
@chef_required
def result(pred_id):
    """
    Affiche le résultat détaillé d'une prédiction pour le chef de service.
    Contrairement au médecin, le chef peut consulter n'importe quelle prédiction.
    Si l'explicabilité a déjà été calculée, génère l'interprétation textuelle.
    Le template doctor/result.html est partagé mais adapté selon le rôle (is_chef).
    """
    # Le chef accède à toutes les prédictions (pas de filtre doctor_id)
    pred = Prediction.query.get_or_404(pred_id)

    # Génération de l'interprétation textuelle si l'explicabilité est disponible
    interpretation = None
    if pred.gradcam_filename:
        interpretation = _generate_interpretation(pred, {
            'perturb_drop_gradcam': pred.perturb_drop_gradcam,
            'perturb_drop_shap':    pred.perturb_drop_shap,
            'perturb_drop_hybrid':  pred.perturb_drop_hybrid,
        })

    return render_template('doctor/result.html', pred=pred, interpretation=interpretation)


# =============================================================================
#  Routes — Explicabilité AJAX (chef de service)
# =============================================================================
@admin_bp.route('/explainability/<int:pred_id>', methods=['POST'])
@chef_required
def compute_explainability(pred_id):
    """
    Endpoint AJAX appelé depuis result.html pour calculer à la demande
    les cartes GradCAM++, SHAP, Hybrid et Segmentation (accès chef de service).
    Si l'explicabilité a déjà été calculée, renvoie les données en cache (DB).
    Retourne un JSON avec les noms de fichiers, les métriques et l'interprétation.
    """
    pred = Prediction.query.get_or_404(pred_id)

    # ── Cache : explicabilité déjà calculée ───────────────────────────────────
    if pred.gradcam_filename:
        return jsonify({
            'gradcam_filename'     : pred.gradcam_filename,
            'shap_filename'        : pred.shap_filename,
            'hybrid_filename'      : pred.hybrid_filename,
            'segmentation_filename': pred.segmentation_filename,
            'perturb_drop_gradcam' : pred.perturb_drop_gradcam,
            'perturb_drop_shap'    : pred.perturb_drop_shap,
            'perturb_drop_hybrid'  : pred.perturb_drop_hybrid,
            'interpretation'       : _generate_interpretation(pred, {
                'perturb_drop_gradcam': pred.perturb_drop_gradcam,
                'perturb_drop_shap':    pred.perturb_drop_shap,
                'perturb_drop_hybrid':  pred.perturb_drop_hybrid,
            }),
        })

    # ── Calcul de l'explicabilité ─────────────────────────────────────────────
    expl_dir   = current_app.config['EXPLAINABILITY_DIR']
    image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], pred.image_filename)

    # Injecter le répertoire de sortie dans les modules ML
    _inject_expl_dir(expl_dir)

    try:
        expl = _run_explainability_only(image_path, pred.model_used, pred.method_used)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # ── Persistance des résultats en base ─────────────────────────────────────
    pred.gradcam_filename      = expl.get('gradcam_filename')
    pred.shap_filename         = expl.get('shap_filename')
    pred.hybrid_filename       = expl.get('hybrid_filename')
    pred.segmentation_filename = expl.get('segmentation_filename')
    pred.perturb_drop_gradcam  = expl.get('perturb_drop_gradcam')
    pred.perturb_drop_shap     = expl.get('perturb_drop_shap')
    pred.perturb_drop_hybrid   = expl.get('perturb_drop_hybrid')
    db.session.commit()

    # ── Génération de l'interprétation textuelle ──────────────────────────────
    expl['interpretation'] = _generate_interpretation(pred, expl)
    return jsonify(expl)


# =============================================================================
#  Routes — Ré-entraînement manuel
# =============================================================================
@admin_bp.route('/retrain', methods=['POST'])
@chef_required
def manual_retrain():
    """
    Déclenche manuellement le ré-entraînement des modèles IA depuis le
    tableau de bord du chef de service. Le ré-entraînement s'exécute dans
    un thread daemon pour ne pas bloquer la réponse HTTP.
    Le statut est tracé dans la table RetrainingLog.
    """
    disease_name = request.form.get('disease_name', '').strip()
    app          = current_app._get_current_object()  # référence à l'app Flask

    # Créer un log de ré-entraînement (statut initial : pending)
    log = RetrainingLog(
        sample_count = FewShotSample.query.count(),
        triggered_by = current_user.id,
        status       = 'pending',
        notes        = (
            f'Manuel — classe : {disease_name}' if disease_name
            else 'Manuel — toutes classes'
        ),
    )
    db.session.add(log)
    db.session.commit()
    log_id = log.id  # récupérer l'ID avant de sortir du contexte

    def run():
        """Fonction exécutée dans le thread d'arrière-plan."""
        with app.app_context():
            log        = RetrainingLog.query.get(log_id)
            log.status = 'running'
            db.session.commit()

            errors        = []
            per_class_dir = app.config['FEW_SHOT_PER_CLASS_FOLDER']

            # ── Ré-entraînement EfficientNet ──────────────────────────────────
            try:
                eff_retrain(per_class_dir)
            except Exception as e:
                errors.append(f'EfficientNet: {e}')

            # ── Ré-entraînement ViT ───────────────────────────────────────────
            try:
                vit_retrain(per_class_dir)
            except Exception as e:
                errors.append(f'ViT: {e}')

            # ── Mise à jour du statut final ───────────────────────────────────
            if errors:
                log.status = 'failed'
                log.notes  = (log.notes or '') + ' | Erreurs : ' + ' ; '.join(errors)
            else:
                log.status = 'completed'

            log.completed_at = datetime.datetime.utcnow()
            db.session.commit()

    # Lancer le thread en mode daemon
    threading.Thread(target=run, daemon=True).start()
    flash('Ré-entraînement lancé en arrière-plan. Vérifiez le statut dans quelques minutes.', 'info')
    return redirect(url_for('admin.dashboard'))


# =============================================================================
#  Routes — Gestion des comptes utilisateurs
# =============================================================================
@admin_bp.route('/accounts')
@chef_required
def accounts():
    """
    Liste tous les comptes médecins et chefs de service.
    Permet au chef de créer, désactiver, réactiver et réinitialiser les mots de passe.
    """
    medecins = User.query.filter_by(role='medecin').order_by(User.created_at.desc()).all()
    chefs    = User.query.filter_by(role='chef_service').order_by(User.created_at.desc()).all()
    return render_template('admin/accounts.html', medecins=medecins, chefs=chefs)


@admin_bp.route('/accounts/create', methods=['POST'])
@chef_required
def create_account():
    """
    Crée un nouveau compte utilisateur (médecin ou chef de service).
    Valide l'unicité du nom d'utilisateur et de l'email avant la création.
    """
    username = request.form.get('username', '').strip()
    email    = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '').strip()
    role     = request.form.get('role', 'medecin')

    # Validation du rôle
    if role not in ('medecin', 'chef_service'):
        flash('Rôle invalide.', 'danger')
        return redirect(url_for('admin.accounts'))

    # Validation des champs obligatoires
    if not username or not email or not password:
        flash('Tous les champs sont obligatoires.', 'danger')
        return redirect(url_for('admin.accounts'))

    # Vérification de l'unicité du nom d'utilisateur
    if User.query.filter_by(username=username).first():
        flash(f'Le nom d\'utilisateur « {username} » est déjà pris.', 'danger')
        return redirect(url_for('admin.accounts'))

    # Vérification de l'unicité de l'email
    if User.query.filter_by(email=email).first():
        flash(f'L\'email « {email} » est déjà utilisé.', 'danger')
        return redirect(url_for('admin.accounts'))

    # Création du compte
    user = User(
        username   = username,
        email      = email,
        role       = role,
        created_by = current_user.id,
    )
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    role_label = 'médecin' if role == 'medecin' else 'chef de service'
    flash(f'Compte {role_label} « {username} » créé avec succès.', 'success')
    return redirect(url_for('admin.accounts'))


@admin_bp.route('/accounts/delete/<int:user_id>', methods=['POST'])
@chef_required
def delete_account(user_id):
    """
    Désactive un compte médecin (suppression logique, pas physique).
    Refuse la désactivation du compte courant et des comptes chefs de service.
    """
    user = User.query.get_or_404(user_id)

    # Protection : ne pas se supprimer soi-même
    if user.id == current_user.id:
        flash('Vous ne pouvez pas supprimer votre propre compte.', 'danger')
        return redirect(url_for('admin.accounts'))

    # Protection : les chefs de service ne peuvent pas être désactivés depuis cette interface
    if user.role == 'chef_service':
        flash('Suppression d\'un chef de service non autorisée depuis cette interface.', 'danger')
        return redirect(url_for('admin.accounts'))

    # Désactivation logique (is_active_account = False)
    user.is_active_account = False
    db.session.commit()
    flash(f'Compte « {user.username} » désactivé.', 'success')
    return redirect(url_for('admin.accounts'))


@admin_bp.route('/accounts/restore/<int:user_id>', methods=['POST'])
@chef_required
def restore_account(user_id):
    """Réactive un compte médecin précédemment désactivé."""
    user = User.query.get_or_404(user_id)
    user.is_active_account = True
    db.session.commit()
    flash(f'Compte « {user.username} » réactivé.', 'success')
    return redirect(url_for('admin.accounts'))


@admin_bp.route('/accounts/reset-password/<int:user_id>', methods=['POST'])
@chef_required
def reset_password(user_id):
    """
    Réinitialise le mot de passe d'un compte utilisateur.
    Exige un minimum de 6 caractères.
    """
    user    = User.query.get_or_404(user_id)
    new_pwd = request.form.get('new_password', '').strip()

    if len(new_pwd) < 6:
        flash('Le mot de passe doit contenir au moins 6 caractères.', 'danger')
        return redirect(url_for('admin.accounts'))

    user.set_password(new_pwd)
    db.session.commit()
    flash(f'Mot de passe de « {user.username} » réinitialisé avec succès.', 'success')
    return redirect(url_for('admin.accounts'))


# =============================================================================
#  Routes — Historique global des analyses
# =============================================================================
@admin_bp.route('/history')
@chef_required
def history():
    """
    Affiche l'historique paginé de toutes les analyses (tous médecins confondus).
    Filtres disponibles : médecin, modèle IA, statut.
    25 résultats par page, triés par date décroissante.
    """
    page          = request.args.get('page', 1, type=int)
    doctor_filter = request.args.get('doctor', 0, type=int)
    status_filter = request.args.get('status', '')
    model_filter  = request.args.get('model', '')

    # Construction de la requête avec filtres optionnels
    query = Prediction.query.order_by(Prediction.created_at.desc())

    if doctor_filter:
        query = query.filter_by(doctor_id=doctor_filter)
    if model_filter:
        query = query.filter_by(model_used=model_filter)

    # Filtre par statut (pending = validated IS NULL)
    if status_filter == 'pending':
        query = query.filter(Prediction.validated.is_(None))
    elif status_filter == 'validated':
        query = query.filter_by(validated=True)
    elif status_filter == 'corrected':
        query = query.filter_by(validated=False)
    elif status_filter == 'unknown':
        query = query.filter_by(is_known=False)

    predictions = query.paginate(page=page, per_page=25, error_out=False)
    medecins    = User.query.filter_by(role='medecin').all()

    return render_template(
        'admin/history.html',
        predictions=predictions,
        medecins=medecins,
        doctor_filter=doctor_filter,
        status_filter=status_filter,
        model_filter=model_filter,
    )


# =============================================================================
#  Helpers internes — construction de la requête filtrée (DRY)
# =============================================================================
def _build_filtered_query(doctor_filter: int, status_filter: str, model_filter: str):
    """
    Construit et retourne une requête SQLAlchemy filtrée pour l'export.
    Factorise la logique commune à export_csv, export_pdf et history.

    Paramètres :
        doctor_filter : 0 = tous, int > 0 = filtrer par doctor_id
        status_filter : '' | 'pending' | 'validated' | 'corrected' | 'unknown'
        model_filter  : '' | 'vit' | 'efficient'
    """
    query = Prediction.query.order_by(Prediction.created_at.desc())

    if doctor_filter:
        query = query.filter_by(doctor_id=doctor_filter)
    if model_filter:
        query = query.filter_by(model_used=model_filter)

    if status_filter == 'pending':
        query = query.filter(Prediction.validated.is_(None))
    elif status_filter == 'validated':
        query = query.filter_by(validated=True)
    elif status_filter == 'corrected':
        query = query.filter_by(validated=False)
    elif status_filter == 'unknown':
        query = query.filter_by(is_known=False)

    return query


# =============================================================================
#  Routes — Export CSV
# =============================================================================
@admin_bp.route('/history/export/csv')
@chef_required
def export_csv():
    """
    Génère et télécharge un fichier CSV de l'historique des analyses
    en respectant les filtres actifs (médecin, statut, modèle).
    Encodage UTF-8 avec BOM pour compatibilité Excel.
    """
    doctor_filter = request.args.get('doctor', 0, type=int)
    status_filter = request.args.get('status', '')
    model_filter  = request.args.get('model', '')

    # Récupération des données filtrées
    rows = _build_filtered_query(doctor_filter, status_filter, model_filter).all()

    # Génération du CSV en mémoire
    output = io.StringIO()
    writer = csv.writer(output)

    # En-têtes du CSV
    writer.writerow([
        'ID', 'Médecin', 'Modèle', 'Méthode', 'Prédiction IA',
        'Confiance (%)', 'OOD', 'Distance OOD', 'Seuil OOD',
        'Diagnostic final', 'Statut', 'Date',
    ])

    # Données — confiance stockée en % (0-100)
    for p in rows:
        writer.writerow([
            p.id,
            p.doctor.username if p.doctor else '',
            p.model_used  or '',
            p.method_used or '',
            p.predicted_disease or 'Inconnu',
            round(p.confidence, 2) if p.confidence else '',
            'OOD' if not p.is_known else 'In-dist.',
            round(p.ood_distance, 4)  if p.ood_distance  else '',
            round(p.ood_threshold, 4) if p.ood_threshold else '',
            p.final_disease or '',
            p.status_label,
            p.created_at.strftime('%d/%m/%Y %H:%M') if p.created_at else '',
        ])

    filename = f"pneumoia_historique_{datetime.date.today().strftime('%Y%m%d')}.csv"
    return Response(
        output.getvalue(),
        mimetype='text/csv; charset=utf-8',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


# =============================================================================
#  Routes — Export PDF
# =============================================================================
@admin_bp.route('/history/export/pdf')
@chef_required
def export_pdf():
    """
    Génère et télécharge un fichier PDF de l'historique des analyses
    en respectant les filtres actifs (médecin, statut, modèle).
    Utilise la bibliothèque fpdf2. Affiche un message d'erreur si non installée.
    Format : paysage A4, en-tête bleu CHU Oran (#29ABE2), lignes alternées.
    """
    # Vérification de la disponibilité de fpdf2
    if not FPDF_AVAILABLE:
        flash('La bibliothèque fpdf2 n\'est pas installée. Exécutez : pip install fpdf2', 'danger')
        return redirect(url_for('admin.history'))

    doctor_filter = request.args.get('doctor', 0, type=int)
    status_filter = request.args.get('status', '')
    model_filter  = request.args.get('model', '')

    # Récupération des données filtrées
    rows = _build_filtered_query(doctor_filter, status_filter, model_filter).all()

    # ── Construction du PDF ───────────────────────────────────────────────────
    pdf = FPDF(orientation='L', format='A4')  # paysage pour plus de colonnes
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Titre du document
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(
        0, 10, 'PneumoIA - CHU Oran - Historique des analyses',
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C',
    )

    # Sous-titre avec date et nombre d'enregistrements
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(
        0, 6,
        f"Exporté le {datetime.date.today().strftime('%d/%m/%Y')} — {len(rows)} enregistrement(s)",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C',
    )
    pdf.ln(4)

    # ── En-têtes du tableau ───────────────────────────────────────────────────
    headers = ['ID', 'Medecin', 'Modele', 'Prediction IA', 'Confiance', 'OOD', 'Diagnostic final', 'Statut', 'Date']
    col_w   = [12,   40,        22,       48,               22,          18,    48,                  22,       28]

    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(41, 171, 226)    # bleu CHU Oran
    pdf.set_text_color(255, 255, 255)   # texte blanc sur fond bleu
    for h, w in zip(headers, col_w):
        pdf.cell(w, 8, h, border=1, fill=True)
    pdf.ln()

    # ── Lignes de données (fond alterné pour lisibilité) ─────────────────────
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(0, 0, 0)

    for i, p in enumerate(rows):
        # Alternance de couleur de fond : bleu très clair / blanc
        if i % 2 == 0:
            pdf.set_fill_color(224, 244, 251)   # bleu très clair
        else:
            pdf.set_fill_color(255, 255, 255)   # blanc

        # Confiance stockée en % (0-100) — 
        conf_str = f"{round(p.confidence, 1)}%" if p.confidence else '-'

        vals = [
            str(p.id),
            (p.doctor.username if p.doctor else '')[:20],
            (p.model_used or '').upper(),
            (p.predicted_disease or 'Inconnu')[:25],
            conf_str,
            'OOD' if not p.is_known else 'In-dist.',
            (p.final_disease or '-')[:25],
            (p.status_label or '-'),
            p.created_at.strftime('%d/%m/%Y %H:%M') if p.created_at else '',
        ]
        for val, w in zip(vals, col_w):
            pdf.cell(w, 7, str(val), border=1, fill=True)
        pdf.ln()

    # ── Génération du fichier et envoi ────────────────────────────────────────
    filename  = f"pneumoia_historique_{datetime.date.today().strftime('%Y%m%d')}.pdf"
    pdf_bytes = pdf.output()
    return Response(
        bytes(pdf_bytes),
        mimetype='application/pdf',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


# =============================================================================
#  Routes — Import CSV
# =============================================================================
@admin_bp.route('/history/import', methods=['POST'])
@chef_required
def import_history():
    """
    Importe l'historique d'analyses depuis un fichier CSV.
    Le CSV doit avoir le même format que l'export PneumoIA.
    Les lignes dont le médecin n'existe pas en base sont ignorées.
    Seules les données texte/numériques sont importées (pas les images).
    """
    f = request.files.get('csv_file')
    if not f or not f.filename.lower().endswith('.csv'):
        flash('Veuillez importer un fichier CSV valide.', 'danger')
        return redirect(url_for('admin.history'))

    # Décodage UTF-8 avec gestion du BOM (utf-8-sig pour compatibilité Excel)
    stream   = io.StringIO(f.stream.read().decode('utf-8-sig'))
    reader   = csv.DictReader(stream)
    imported = 0
    errors   = 0

    for row in reader:
        try:
            # Récupération du médecin par nom d'utilisateur
            doctor_name = row.get('Médecin') or row.get('Medecin', '')
            doctor      = User.query.filter_by(username=doctor_name.strip()).first()
            if not doctor:
                # Médecin introuvable → ignorer la ligne
                errors += 1
                continue

            # Création de la prédiction importée
            # Note : image_filename = 'imported.png' (pas de fichier image réel)
            p = Prediction(
                doctor_id         = doctor.id,
                model_used        = (row.get('Modèle') or row.get('Modele', '')).strip().lower() or 'vit',
                method_used       = row.get('Méthode') or row.get('Methode') or None,
                predicted_disease = row.get('Prédiction IA') or row.get('Prediction IA') or None,
                confidence        = float(row['Confiance (%)']) if row.get('Confiance (%)') else None,
                is_known          = (row.get('OOD', 'In-dist.').strip().lower() == 'in-dist.'),
                final_disease     = row.get('Diagnostic final') or None,
                image_filename    = 'imported.png',
            )
            db.session.add(p)
            imported += 1

        except Exception:
            # En cas d'erreur de parsing → ignorer la ligne silencieusement
            errors += 1

    db.session.commit()
    flash(
        f'{imported} enregistrement(s) importé(s). {errors} ligne(s) ignorée(s).',
        'success' if imported else 'warning',
    )
    return redirect(url_for('admin.history'))


# =============================================================================
#  Routes — Page des validations
# =============================================================================
@admin_bp.route('/validations')
@chef_required
def validations():
    """
    Affiche la liste paginée des prédictions pour validation par le chef de service.
    Filtres disponibles : médecin, statut.
    20 résultats par page, triés par date décroissante.
    Chaque ligne est cliquable pour accéder au détail de la prédiction.
    """
    page          = request.args.get('page', 1, type=int)
    doctor_filter = request.args.get('doctor', 0, type=int)
    status_filter = request.args.get('status', '')

    # Construction de la requête avec filtres
    query = Prediction.query.order_by(Prediction.created_at.desc())

    if doctor_filter:
        query = query.filter_by(doctor_id=doctor_filter)

    if status_filter == 'pending':
        query = query.filter(Prediction.validated.is_(None))
    elif status_filter == 'validated':
        query = query.filter_by(validated=True)
    elif status_filter == 'corrected':
        query = query.filter_by(validated=False)
    elif status_filter == 'unknown':
        query = query.filter_by(is_known=False)

    predictions = query.paginate(page=page, per_page=20, error_out=False)
    medecins    = User.query.filter_by(role='medecin').all()

    return render_template(
        'admin/validations.html',
        predictions=predictions,
        medecins=medecins,
        doctor_filter=doctor_filter,
        status_filter=status_filter,
    )
