# =============================================================================
#  routes/doctor.py
#  Blueprint Flask pour les médecins — upload, résultats, historique,
#  explicabilité AJAX, few-shot et ré-entraînement automatique.
# =============================================================================

# ─── Bibliothèques standard Python ───────────────────────────────────────────
import os           # manipulation des chemins de fichiers
import math         # vérification inf/nan pour la sérialisation JSON
import shutil       # copie de fichiers (few-shot per-class)
import uuid         # génération de noms de fichiers uniques
import threading    # ré-entraînement asynchrone en arrière-plan
from datetime import datetime    # horodatage des logs de ré-entraînement
from functools import wraps      # décorateur d'accès @medecin_required

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
)
from flask_login import login_required, current_user

# ─── Extensions et modèles locaux ─────────────────────────────────────────────
from extensions import db
from models import Prediction, FewShotSample, RetrainingLog

# ─── Modules ML (chargés une seule fois au démarrage de l'application) ────────
     
from ml import vit_model, efficient_model                    # modules complets (set_explainability_dir)
from ml.vit_model import (
    predict                    as vit_predict,               # inférence ViT
    compute_explainability     as vit_compute_explainability, # GradCAM++/SHAP ViT
    few_shot_compare           as vit_few_shot_compare,      # comparaison few-shot ViT
    retrain                    as vit_retrain,               # ré-entraînement ViT
)
from ml.efficient_model import (
    predict_energy             as eff_predict_energy,        # inférence EfficientNet (méthode Energy)
    predict_mahalanobis        as eff_predict_mahalanobis,   # inférence EfficientNet (méthode Mahalanobis)
    compute_explainability     as eff_compute_explainability, # GradCAM++/SHAP EfficientNet
    few_shot_compare           as eff_few_shot_compare,      # comparaison few-shot EfficientNet
    retrain                    as eff_retrain,               # ré-entraînement EfficientNet
)

# =============================================================================
#  Blueprint
# =============================================================================
doctor_bp = Blueprint('doctor', __name__)

# Extensions autorisées pour l'upload d'images radiologiques
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'dcm'}


# =============================================================================
#  Décorateur d'accès — médecin uniquement
# =============================================================================
def medecin_required(f):
    """
    Décorateur qui restreint l'accès aux utilisateurs ayant le rôle 'medecin'.
    Redirige vers le tableau de bord admin si l'utilisateur est chef de service.
    """
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if not current_user.is_medecin:
            flash('Accès réservé aux médecins.', 'danger')
            return redirect(url_for('admin.dashboard'))
        return f(*args, **kwargs)
    return decorated


# =============================================================================
#  Helpers fichiers
# =============================================================================
def allowed_file(filename: str) -> bool:
    """Vérifie que l'extension du fichier est dans ALLOWED_EXTENSIONS."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def secure_unique_filename(original_filename: str) -> str:
    """
    Génère un nom de fichier unique basé sur UUID pour éviter les collisions
    et les injections de chemin (path traversal).
    """
    ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'png'
    return f"{uuid.uuid4().hex}.{ext}"


# =============================================================================
#  Routes — Tableau de bord médecin
# =============================================================================
@doctor_bp.route('/dashboard')
@medecin_required
def dashboard():
    """
    Page principale du médecin.
    Affiche les statistiques personnelles (total, validées, inconnues, corrigées),
    la répartition des maladies détectées et les analyses récentes.
    """
    # Toutes les prédictions du médecin, triées par date décroissante
    recent = (
        Prediction.query
        .filter_by(doctor_id=current_user.id)
        .order_by(Prediction.created_at.desc())
        .all()
    )

    # Compteurs pour les cartes de statistiques
    total           = Prediction.query.filter_by(doctor_id=current_user.id).count()
    validated_count = Prediction.query.filter_by(doctor_id=current_user.id, validated=True).count()
    unknown_count   = Prediction.query.filter_by(doctor_id=current_user.id, is_known=False).count()

    # Corrigées = médecin a fourni le bon diagnostic (validated=False + is_known=True + final_disease renseigné)
    corrected_count = (
        Prediction.query
        .filter_by(doctor_id=current_user.id, validated=False, is_known=True)
        .filter(Prediction.final_disease.isnot(None))
        .count()
    )

    # Répartition par maladie prédite (uniquement les cas connus)
    rows = (
        db.session.query(Prediction.predicted_disease, func.count(Prediction.id))
        .filter(
            Prediction.doctor_id == current_user.id,
            Prediction.is_known  == True,
            Prediction.predicted_disease.isnot(None),
        )
        .group_by(Prediction.predicted_disease)
        .order_by(func.count(Prediction.id).desc())
        .all()
    )
    disease_counts = {disease: count for disease, count in rows}

    return render_template(
        'doctor/dashboard.html',
        recent_predictions=recent,
        total=total,
        validated_count=validated_count,
        unknown_count=unknown_count,
        corrected_count=corrected_count,
        disease_counts=disease_counts,
    )


# =============================================================================
#  Routes — Upload et analyse
# =============================================================================
@doctor_bp.route('/upload', methods=['GET', 'POST'])
@medecin_required
def upload():
    """
    Page d'upload d'une nouvelle radiographie.
    GET  → affiche le formulaire.
    POST → valide le fichier, lance l'inférence ML et redirige vers le résultat.
    """
    if request.method == 'POST':

        # ── Validation du fichier ────────────────────────────────────────────
        if 'image' not in request.files or request.files['image'].filename == '':
            flash('Veuillez sélectionner une image radiographique.', 'danger')
            return redirect(url_for('doctor.upload'))

        file = request.files['image']
        if not allowed_file(file.filename):
            flash('Format non supporté. Utilisez PNG, JPG, BMP, TIFF ou DCM.', 'danger')
            return redirect(url_for('doctor.upload'))

        # ── Validation du choix de modèle ────────────────────────────────────
        model_choice = request.form.get('model')
        if model_choice not in ('vit', 'efficient'):
            flash('Veuillez sélectionner un modèle.', 'danger')
            return redirect(url_for('doctor.upload'))

        # EfficientNet nécessite une méthode OOD supplémentaire
        method_choice = None
        if model_choice == 'efficient':
            method_choice = request.form.get('method')
            if method_choice not in ('mahalanobis', 'energy'):
                flash('Veuillez sélectionner une méthode pour EfficientNet.', 'danger')
                return redirect(url_for('doctor.upload'))

        # ── Sauvegarde de l'image sur disque ─────────────────────────────────
        filename   = secure_unique_filename(file.filename)
        image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        # ── Injecter le répertoire d'explicabilité dans les modules ML ────────
        expl_dir = current_app.config['EXPLAINABILITY_DIR']
        _inject_expl_dir(expl_dir)

        # ── Lancement de l'inférence ──────────────────────────────────────────
        try:
            result = _run_inference(image_path, model_choice, method_choice)
        except Exception as e:
            flash(f'Erreur lors de l\'analyse : {e}', 'danger')
            return redirect(url_for('doctor.upload'))

        # ── Persistance en base de données ────────────────────────────────────
        pred = Prediction(
            doctor_id              = current_user.id,
            image_filename         = filename,
            model_used             = model_choice,
            method_used            = method_choice,
            predicted_disease      = result.get('disease'),
            confidence             = result.get('confidence'),
            is_known               = result.get('is_known', True),
            ood_distance           = result.get('ood_distance'),
            ood_threshold          = result.get('ood_threshold'),
            gradcam_filename       = result.get('gradcam_filename'),
            shap_filename          = result.get('shap_filename'),
            hybrid_filename        = result.get('hybrid_filename'),
            segmentation_filename  = result.get('segmentation_filename'),
            perturb_drop_gradcam   = result.get('perturb_drop_gradcam'),
            perturb_drop_shap      = result.get('perturb_drop_shap'),
            perturb_drop_hybrid    = result.get('perturb_drop_hybrid'),
            sparsity_score         = result.get('sparsity_score'),
            focus_score            = result.get('focus_score'),
        )
        db.session.add(pred)
        db.session.commit()

        return redirect(url_for('doctor.result', pred_id=pred.id))

    # GET → afficher le formulaire d'upload
    return render_template('doctor/upload.html')


# =============================================================================
#  Routes — Résultat d'une analyse
# =============================================================================
@doctor_bp.route('/result/<int:pred_id>')
@medecin_required
def result(pred_id):
    """
    Affiche le résultat détaillé d'une analyse pour le médecin propriétaire.
    Si l'explicabilité a déjà été calculée, génère l'interprétation textuelle.
    """
    # Sécurité : le médecin ne peut voir que ses propres prédictions
    pred = Prediction.query.filter_by(id=pred_id, doctor_id=current_user.id).first_or_404()

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
#  Routes — Validation d'une prédiction
# =============================================================================
@doctor_bp.route('/validate/<int:pred_id>', methods=['POST'])
@medecin_required
def validate(pred_id):
    """
    Permet au médecin de valider ou corriger une prédiction IA.
    - action='correct'            → valide le diagnostic IA tel quel.
    - action='correct_with_name'  → remplace le diagnostic par celui du médecin.
    """
    pred = Prediction.query.filter_by(id=pred_id, doctor_id=current_user.id).first_or_404()

    # Vérification : une prédiction déjà traitée ne peut plus être modifiée
    if pred.validated is not None:
        flash('Cette prédiction a déjà été traitée.', 'warning')
        return redirect(url_for('doctor.result', pred_id=pred_id))

    action = request.form.get('action')

    if action == 'correct':
        # Le médecin confirme le diagnostic de l'IA
        pred.validated     = True
        pred.is_known      = True
        pred.final_disease = pred.predicted_disease
        db.session.commit()
        flash(f'Prédiction validée : {pred.final_disease}.', 'success')

    elif action == 'correct_with_name':
        # Le médecin fournit le bon diagnostic
        corrected = request.form.get('corrected_disease', '').strip()
        if not corrected:
            flash('Veuillez saisir le nom de la maladie correcte.', 'danger')
            return redirect(url_for('doctor.result', pred_id=pred_id))
        pred.validated         = False
        pred.is_known          = True
        pred.predicted_disease = corrected
        pred.final_disease     = corrected
        db.session.commit()
        flash(f'Diagnostic corrigé : {pred.final_disease}.', 'info')

    return redirect(url_for('doctor.dashboard'))


# =============================================================================
#  Routes — Soumission d'un cas inconnu (OOD) à la base few-shot
# =============================================================================
@doctor_bp.route('/unknown-disease/<int:pred_id>', methods=['POST'])
@medecin_required
def unknown_disease(pred_id):
    """
    Enregistre une image OOD dans la base few-shot avec le nom de maladie
    fourni par le médecin. Copie également l'image dans le dossier per-class
    utilisé par la comparaison few-shot.
    Déclenche un ré-entraînement automatique si le seuil par classe est atteint.
    """
    pred = Prediction.query.filter_by(id=pred_id, doctor_id=current_user.id).first_or_404()

    # Vérification : prédiction déjà traitée
    if pred.validated is not None:
        flash('Cette prédiction a déjà été traitée.', 'warning')
        return redirect(url_for('doctor.result', pred_id=pred_id))

    disease_name = request.form.get('disease_name', '').strip()
    if not disease_name:
        flash('Veuillez saisir le nom de la maladie.', 'danger')
        return redirect(url_for('doctor.result', pred_id=pred_id))

    # Mise à jour de la prédiction
    pred.final_disease = disease_name
    pred.validated     = False

    # Insertion dans la table FewShotSample (source de vérité)
    few_shot_db = FewShotSample(
        prediction_id  = pred.id,
        disease_name   = disease_name,
        image_filename = pred.image_filename,
        method         = pred.method_used or 'energy',
    )
    db.session.add(few_shot_db)
    db.session.commit()

    # Copie de l'image dans le dossier per-class (pour la comparaison few-shot)
    per_class_dir = current_app.config['FEW_SHOT_PER_CLASS_FOLDER']
    cls_dir       = os.path.join(per_class_dir, disease_name.replace('/', '_'))
    os.makedirs(cls_dir, exist_ok=True)
    src = os.path.join(current_app.config['UPLOAD_FOLDER'], pred.image_filename)
    dst = os.path.join(cls_dir, pred.image_filename)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

    # ── Recalcul immédiat des features few-shot ───────────────────────────────
    # Après chaque ajout d'image, on pre-calcule les features de la nouvelle
    # image dans le cache mémoire. Ainsi, la PROCHAINE comparaison few-shot
    # tiendra automatiquement compte de cette nouvelle image sans délai.
    try:
        _run_few_shot(
            src,                                                  # image venant d'être ajoutée
            pred.model_used,
            pred.method_used,
            per_class_dir,
            current_app.config.get('FEW_SHOT_SIM_THRESHOLD', 0.72),
        )
    except Exception:
        # Le recalcul sera effectué automatiquement lors de la prochaine comparaison
        pass

    # ── Comptage par classe et vérification du seuil ──────────────────────────
    cls_count      = FewShotSample.query.filter_by(disease_name=disease_name).count()
    per_cls_thresh = current_app.config['FEW_SHOT_PER_CLASS_THRESHOLD']

    if cls_count >= per_cls_thresh:
        # Seuil atteint → notifier le chef de service pour qu'il lance le ré-entraînement
        flash(
            f'Maladie « {disease_name} » : seuil de {per_cls_thresh} images atteint '
            f'({cls_count} images). Le chef de service peut lancer le ré-entraînement '
            f'depuis son tableau de bord.',
            'warning',
        )
    else:
        # Seuil non encore atteint → informer le médecin de la progression
        flash(
            f'Maladie « {disease_name} » ajoutée à la base few-shot '
            f'({cls_count}/{per_cls_thresh} images pour cette classe). '
            f'Features mises à jour — cette image sera prise en compte dès la prochaine comparaison.',
            'success',
        )

    return redirect(url_for('doctor.dashboard'))


# =============================================================================
#  Routes — Explicabilité AJAX (médecin)
# =============================================================================
@doctor_bp.route('/explainability/<int:pred_id>', methods=['POST'])
@medecin_required
def compute_explainability(pred_id):
    """
    Endpoint AJAX appelé depuis result.html pour calculer à la demande
    les cartes GradCAM++, SHAP, Hybrid et Segmentation.
    Si l'explicabilité a déjà été calculée, renvoie les données en cache (DB).
    Retourne un JSON avec les noms de fichiers, les métriques de perturbation
    et l'interprétation textuelle générée automatiquement.
    """
    # Sécurité : le médecin ne peut accéder qu'à ses propres prédictions
    pred = Prediction.query.filter_by(id=pred_id, doctor_id=current_user.id).first_or_404()

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
#  Routes — Comparaison few-shot AJAX
# =============================================================================
@doctor_bp.route('/few-shot/<int:pred_id>', methods=['POST'])
@medecin_required
def few_shot(pred_id):
    """
    Endpoint AJAX : compare l'image OOD avec la base few-shot per-class.
    Retourne la maladie la plus proche et le score de similarité.
    Si une maladie est trouvée, met à jour predicted_disease dans la DB
    (is_known reste False jusqu'à validation explicite du médecin).
    """
    pred = Prediction.query.filter_by(id=pred_id, doctor_id=current_user.id).first_or_404()

    few_shot_dir = current_app.config['FEW_SHOT_PER_CLASS_FOLDER']
    image_path   = os.path.join(current_app.config['UPLOAD_FOLDER'], pred.image_filename)
    threshold    = current_app.config.get('FEW_SHOT_SIM_THRESHOLD', 0.72)

    try:
        result = _run_few_shot(image_path, pred.model_used, pred.method_used, few_shot_dir, threshold)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Si une maladie proche a été trouvée → stocker la suggestion (sans valider)
    if result.get('found') and result.get('disease'):
        pred.predicted_disease = result['disease']
        # is_known reste False : c'est la validation du médecin qui basculera le statut
        db.session.commit()

    # Sanitisation JSON : float('inf') et float('nan') ne sont pas sérialisables
    def _json_safe(v):
        """Remplace inf/nan par None pour une sérialisation JSON correcte."""
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            return None
        return v

    result = {
        k: _json_safe(v) if not isinstance(v, dict) else
           {ck: _json_safe(cv) for ck, cv in v.items()}
        for k, v in result.items()
    }

    return jsonify(result)


# =============================================================================
#  Routes — Historique du médecin
# =============================================================================
@doctor_bp.route('/history')
@medecin_required
def history():
    """
    Affiche l'historique paginé des analyses du médecin connecté.
    15 résultats par page, triés par date décroissante.
    """
    page = request.args.get('page', 1, type=int)
    predictions = (
        Prediction.query
        .filter_by(doctor_id=current_user.id)
        .order_by(Prediction.created_at.desc())
        .paginate(page=page, per_page=15, error_out=False)
    )
    return render_template('doctor/history.html', predictions=predictions)


# =============================================================================
#  Helpers ML — injection du répertoire de sortie
# =============================================================================
def _inject_expl_dir(path: str):
    """
    Injecte le répertoire de sauvegarde des images d'explicabilité
    dans les deux modules ML (ViT et EfficientNet).
    Doit être appelé avant toute génération de cartes GradCAM++/SHAP.
    """
    vit_model.set_explainability_dir(path)
    efficient_model.set_explainability_dir(path)


# =============================================================================
#  Helpers ML — inférence principale
# =============================================================================
def _run_inference(image_path: str, model: str, method: str | None) -> dict:
    """
    Lance l'inférence (prédiction de maladie + score OOD) sans calculer
    l'explicabilité (qui est optionnelle et calculée à la demande).

    Paramètres :
        image_path : chemin absolu vers l'image radiologique
        model      : 'vit' ou 'efficient'
        method     : None (ViT) | 'energy' | 'mahalanobis' (EfficientNet)

    Retourne un dict avec les clés :
        disease, confidence, is_known, ood_distance, ood_threshold,
        gradcam_filename (optionnel), shap_filename (optionnel), ...
    """
    if model == 'vit':
        # ViT : détection OOD intégrée, pas de méthode supplémentaire requise
        return vit_predict(image_path)

    if model == 'efficient':
        if method == 'energy':
            # EfficientNet + score d'énergie pour la détection OOD
            return eff_predict_energy(image_path)
        if method == 'mahalanobis':
            # EfficientNet + distance de Mahalanobis pour la détection OOD
            return eff_predict_mahalanobis(image_path)

    raise ValueError(f'Combinaison inconnue : modèle={model}, méthode={method}')


# =============================================================================
#  Helpers ML — explicabilité à la demande
# =============================================================================
def _run_explainability_only(image_path: str, model: str, method: str | None) -> dict:
    """
    Calcule uniquement les cartes d'explicabilité (GradCAM++, SHAP, Hybrid,
    Segmentation) sans refaire l'inférence. Appelé depuis les endpoints AJAX.

    Retourne un dict avec les clés :
        gradcam_filename, shap_filename, hybrid_filename, segmentation_filename,
        perturb_drop_gradcam, perturb_drop_shap, perturb_drop_hybrid,
        sparsity_score, focus_score
    """
    if model == 'vit':
        return vit_compute_explainability(image_path)
    if model == 'efficient':
        return eff_compute_explainability(image_path)
    # Modèle inconnu → retourner un dictionnaire vide sans lever d'exception
    return {}


# =============================================================================
#  Helpers ML — comparaison few-shot
# =============================================================================
def _run_few_shot(image_path: str, model: str, method: str | None,
                  few_shot_dir: str, threshold: float) -> dict:
    """
    Compare l'image OOD avec la base few-shot per-class pour identifier
    la maladie la plus proche par similarité de features.

    Paramètres :
        image_path   : chemin absolu vers l'image à comparer
        model        : 'vit' ou 'efficient'
        method       : non utilisé (les deux méthodes EfficientNet partagent les features)
        few_shot_dir : dossier racine contenant les sous-dossiers par classe
        threshold    : seuil de similarité cosinus (défaut : 0.72)

    Retourne un dict avec les clés :
        found (bool), disease (str|None), similarity (float), top_classes (list)
    """
    if model == 'vit':
        # ViT : comparaison via embeddings du transformeur
        return vit_few_shot_compare(image_path, few_shot_dir)

    # EfficientNet : les méthodes energy et mahalanobis partagent le même
    # extracteur de features → même comparaison few-shot
    return eff_few_shot_compare(image_path, few_shot_dir)


# =============================================================================
#  Génération de l'interprétation textuelle
# =============================================================================
def _generate_interpretation(pred, expl_data: dict) -> dict:
    """
    Génère une interprétation textuelle dynamique et contextualisée à partir
    des résultats réels de l'analyse (maladie, confiance, drops de perturbation).

    Les textes varient selon 4 niveaux de fiabilité basés sur la chute de
    confiance lors du masquage des zones actives :
        - very_high : chute >= 50 % (zones très discriminantes)
        - high      : chute >= 30 %
        - moderate  : chute >= 20 % (limite de validité)
        - low       : chute <  20 % (insuffisant)

    Paramètres :
        pred      : objet Prediction (modèle SQLAlchemy)
        expl_data : dict avec les clés perturb_drop_gradcam, perturb_drop_shap,
                    perturb_drop_hybrid

    Retourne un dict avec les clés :
        gradcam, shap, hybrid, segmentation, overall
    """
    # ── Données de base ───────────────────────────────────────────────────────
    disease  = (pred.predicted_disease or 'inconnue').strip()
    conf     = round(pred.confidence, 1) if pred.confidence else None
    model    = (pred.model_used or 'vit').upper()
    is_known = pred.is_known

    # Métriques de perturbation (None si non calculé)
    drop_gc = expl_data.get('perturb_drop_gradcam')
    drop_sh = expl_data.get('perturb_drop_shap')
    drop_hy = expl_data.get('perturb_drop_hybrid')

    # ── Fonction utilitaire : niveau de fiabilité ─────────────────────────────
    def _level(drop):
        """Catégorise la chute de confiance en niveau qualitatif."""
        if drop is None: return 'unknown'
        if drop >= 50:   return 'very_high'
        if drop >= 30:   return 'high'
        if drop >= 20:   return 'moderate'
        return 'low'

    def _validity_label(drop):
        """Génère un label textuel court décrivant la fiabilité du drop."""
        if drop is None: return 'non calculé'
        if drop >= 50:   return f'très fiable — chute {round(drop,1)} % (seuil : 20 %)'
        if drop >= 30:   return f'fiable — chute {round(drop,1)} %'
        if drop >= 20:   return f'limite — chute {round(drop,1)} % ≈ seuil 20 %'
        return f'insuffisant — chute {round(drop,1)} % < 20 %'

    # ── Zones anatomiques typiques par maladie ────────────────────────────────
    #    Correspondance par mot-clé (sous-chaîne en minuscules)
    _zones = {
        'pneumoni':    'les lobes inférieurs, avec opacités alvéolaires bilatérales ou unilatérales',
        'covid':       'les zones périphériques et postérieures des deux poumons (aspect en verre dépoli)',
        'tuberculos':  "les lobes supérieurs, avec cavitations et infiltrats nodulaires",
        'pneumothor':  'le bord libre pulmonaire et la plèvre pariétale (absence de trames vasculaires)',
        'effusion':    'les zones basales et costophréniques (opacité déclive homogène)',
        'epanchement': 'les zones basales et costophréniques (opacité déclive homogène)',
        'atelectas':   'les segments pulmonaires collabés (opacité segmentaire ou lobaire)',
        'cardiomeg':   'la silhouette cardiaque élargie (rapport cardio-thoracique > 0,5)',
        'nodule':      'les nodules pulmonaires identifiés',
        'masse':       'la masse pulmonaire et ses rapports avec les structures adjacentes',
        'normal':      "l'ensemble du parenchyme pulmonaire sans anomalie focale",
        'sain':        "l'ensemble du parenchyme pulmonaire sans anomalie focale",
    }
    # Chercher la zone correspondante (défaut si aucune correspondance)
    zone_desc = 'les zones pulmonaires présentant des anomalies radiologiques'
    for key, desc in _zones.items():
        if key in disease.lower():
            zone_desc = desc
            break

    # ── Texte GradCAM++ ───────────────────────────────────────────────────────
    gc_lvl = _level(drop_gc)
    if gc_lvl == 'very_high':
        gradcam_text = (
            f"La carte GradCAM++ confirme de manière très convaincante que le modèle {model} "
            f"a ciblé précisément les zones pathologiques pour classer cette image en « {disease} ». "
            f"La chute de confiance de {round(drop_gc,1)} % lors du masquage — bien au-dessus du "
            f"seuil de 20 % — indique que ces régions sont les véritables déterminants de la décision. "
            f"Zones concernées : {zone_desc}."
        )
    elif gc_lvl == 'high':
        gradcam_text = (
            f"La carte GradCAM++ identifie clairement les régions décisives pour le diagnostic "
            f"de « {disease} » par le modèle {model}. La chute de confiance de {round(drop_gc,1)} % "
            f"après masquage valide que ces zones — {zone_desc} — ont bien guidé la classification. "
            f"L'activation est cohérente avec ce qui est attendu radiologiquement."
        )
    elif gc_lvl == 'moderate':
        gradcam_text = (
            f"La carte GradCAM++ suggère une activation sur {zone_desc} pour « {disease} », "
            f"avec une chute de confiance de {round(drop_gc,1)} % à la limite du seuil de validité (20 %). "
            f"Le modèle {model} a probablement ciblé les bonnes régions, mais d'autres zones "
            f"secondaires ont pu influencer la décision. Une interprétation prudente est recommandée."
        )
    elif gc_lvl == 'low':
        gradcam_text = (
            f"La carte GradCAM++ montre des zones d'activation pour « {disease} », mais la faible "
            f"chute de confiance ({round(drop_gc,1)} % < 20 %) indique que le modèle {model} "
            f"n'a pas focalisé exclusivement sur des régions pathologiques spécifiques. "
            f"L'activation est dispersée sur {zone_desc} et d'autres régions moins pertinentes. "
            f"Ce résultat doit être interprété avec prudence."
        )
    else:
        # Perturbation non calculée
        gradcam_text = (
            f"La carte GradCAM++ visualise les zones ayant influencé la classification "
            f"en « {disease} » par le modèle {model}. Zones attendues : {zone_desc}. "
            f"La perturbation n'a pas encore été calculée pour cette analyse."
        )

    # ── Texte SHAP ────────────────────────────────────────────────────────────
    sh_lvl = _level(drop_sh)

    # Note de cohérence entre GradCAM++ et SHAP
    if drop_gc is not None and drop_sh is not None:
        diff = drop_sh - drop_gc
        if abs(diff) < 8:
            consistency = (
                " Les résultats SHAP sont cohérents avec GradCAM++, "
                "ce qui renforce la fiabilité globale de l'explicabilité."
            )
        elif diff > 0:
            consistency = (
                f" SHAP identifie des zones encore plus discriminantes que GradCAM++ "
                f"(+{round(diff,1)} %), suggérant une contribution pixellaire très localisée."
            )
        else:
            consistency = (
                f" GradCAM++ identifie des zones légèrement plus discriminantes que SHAP "
                f"({round(abs(diff),1)} % d'écart), ce qui est courant pour des patterns diffus."
            )
    else:
        consistency = ""

    if sh_lvl == 'very_high':
        shap_text = (
            f"L'analyse SHAP attribue des contributions très fortes ({round(drop_sh,1)} % de chute) "
            f"aux pixels des zones pathologiques. Les régions en jaune vif représentent les pixels "
            f"les plus déterminants pour la prédiction de « {disease} » — leur masquage effondre "
            f"la confiance du modèle {model} de façon spectaculaire." + consistency
        )
    elif sh_lvl == 'high':
        shap_text = (
            f"L'analyse SHAP (SHapley Additive exPlanations) identifie clairement les pixels "
            f"contributeurs au diagnostic de « {disease} ». La chute de {round(drop_sh,1)} % "
            f"({_validity_label(drop_sh)}) confirme que les zones en jaune vif — situées sur "
            f"{zone_desc} — sont les principaux responsables de la décision." + consistency
        )
    elif sh_lvl == 'moderate':
        shap_text = (
            f"L'analyse SHAP révèle une contribution modérée des zones identifiées pour "
            f"« {disease} » (chute : {round(drop_sh,1)} % — {_validity_label(drop_sh)}). "
            f"Les pixels en jaune ont un impact positif sur la classification, mais la décision "
            f"repose également sur d'autres caractéristiques globales de l'image." + consistency
        )
    elif sh_lvl == 'low':
        shap_text = (
            f"L'analyse SHAP montre une dispersion de l'activation pour « {disease} » "
            f"(chute : {round(drop_sh,1)} % < 20 %). La contribution des zones surlignées est "
            f"relativement faible, ce qui peut indiquer que le modèle {model} s'est appuyé sur "
            f"des caractéristiques globales ou texturales plutôt que sur des lésions focales précises."
            + consistency
        )
    else:
        shap_text = (
            f"L'analyse SHAP quantifie la contribution de chaque pixel à la prédiction "
            f"de « {disease} » par le modèle {model}. Les zones en jaune vif ont un impact "
            f"positif, les zones sombres un impact nul ou négatif." + consistency
        )

    # ── Texte Hybrid ──────────────────────────────────────────────────────────
    hy_lvl = _level(drop_hy)

    # Note de comparaison entre les trois méthodes
    if drop_gc is not None and drop_sh is not None and drop_hy is not None:
        best = max(drop_gc, drop_sh, drop_hy)
        if drop_hy == best:
            fusion_note = (
                f"La fusion Hybrid ({round(drop_hy,1)} %) surpasse GradCAM++ "
                f"({round(drop_gc,1)} %) et SHAP ({round(drop_sh,1)} %) individuellement, "
                f"confirmant que la combinaison des deux méthodes localise mieux les zones pathologiques."
            )
        else:
            fusion_note = (
                f"La carte Hybrid ({round(drop_hy,1)} %) consolide GradCAM++ "
                f"({round(drop_gc,1)} %) et SHAP ({round(drop_sh,1)} %) en supprimant "
                f"les activations parasites non partagées entre les deux méthodes."
            )
    elif drop_hy is not None:
        fusion_note = f"Chute de confiance Hybrid : {round(drop_hy,1)} % ({_validity_label(drop_hy)})."
    else:
        fusion_note = "La perturbation Hybrid n'a pas encore été calculée."

    if hy_lvl in ('very_high', 'high'):
        hybrid_text = (
            f"La carte Hybrid fusionne GradCAM++ et SHAP avec un résultat probant pour "
            f"« {disease} » : seules les zones confirmées par les deux méthodes sont conservées, "
            f"éliminant les faux positifs d'activation. " + fusion_note +
            f" Zones ciblées : {zone_desc}."
        )
    elif hy_lvl == 'moderate':
        hybrid_text = (
            f"La carte Hybrid offre une vue consolidée des zones d'intérêt pour « {disease} », "
            f"à la limite du seuil de fiabilité. " + fusion_note +
            f" Elle confirme l'activation sur {zone_desc}, mais avec une certaine incertitude."
        )
    elif hy_lvl == 'low':
        hybrid_text = (
            f"La carte Hybrid pour « {disease} » présente une faible chute de confiance "
            f"après masquage. " + fusion_note +
            f" Les zones identifiées sur {zone_desc} méritent une vérification clinique."
        )
    else:
        hybrid_text = (
            f"La carte Hybrid fusionne GradCAM++ et SHAP pour réduire les faux positifs "
            f"et confirmer les zones actives pour « {disease} » — {zone_desc}. " + fusion_note
        )

    # ── Texte Segmentation ────────────────────────────────────────────────────
    if not is_known:
        # Image OOD : contexte d'incertitude
        segm_context = (
            f"Cette image étant classée hors-distribution (OOD), la segmentation délimite "
            f"les zones d'activation malgré l'incertitude du modèle sur la classe exacte. "
            f"Le contour cyan doit être interprété avec prudence."
        )
    elif conf and conf >= 85:
        # Confiance élevée : segmentation précise
        segm_context = (
            f"Avec une confiance élevée ({conf} %), la segmentation délimite avec précision "
            f"les contours des zones pathologiques pour « {disease} ». Le contour cyan coïncide "
            f"avec les régions les plus activées et correspond à {zone_desc}."
        )
    elif conf and conf >= 70:
        # Confiance modérée : quelques régions périphériques possibles
        segm_context = (
            f"Avec une confiance modérée ({conf} %), la segmentation identifie les principales "
            f"zones d'intérêt pour « {disease} » ({zone_desc}), mais des régions périphériques "
            f"peuvent aussi être incluses dans le contour."
        )
    elif conf:
        # Confiance faible : segmentation moins précise
        segm_context = (
            f"La confiance relativement faible du modèle ({conf} %) se reflète dans la "
            f"segmentation : le contour cyan peut englober des zones non strictement pathologiques "
            f"pour « {disease} »."
        )
    else:
        segm_context = (
            f"La segmentation délimite les zones activées par le modèle pour « {disease} » "
            f"({zone_desc})."
        )

    segm_text = (
        f"La carte de segmentation superpose un contour cyan sur la carte thermique pour "
        f"localiser anatomiquement les régions concernées : {zone_desc}. "
        + segm_context
        + f" Cette visualisation facilite la corrélation avec les repères radiologiques."
    )

    # ── Bilan global ──────────────────────────────────────────────────────────
    drops       = [d for d in [drop_gc, drop_sh, drop_hy] if d is not None]
    avg_drop    = round(sum(drops) / len(drops), 1) if drops else None
    valid_count = sum(1 for d in drops if d >= 20)   # méthodes au-dessus du seuil
    max_drop    = max(drops) if drops else None

    if not drops:
        # Aucune perturbation calculée
        overall = (
            f"Les cartes d'explicabilité ont été générées pour la prédiction « {disease} » "
            f"par le modèle {model}. Cliquez sur chaque carte pour identifier les zones anatomiques "
            f"d'intérêt. La validation par perturbation n'a pas encore été effectuée."
        )
    elif avg_drop >= 40:
        overall = (
            f"Explicabilité excellente — {valid_count}/{len(drops)} méthodes valides, "
            f"chute moyenne : {avg_drop} % (max : {round(max_drop,1)} %). "
            f"Le modèle {model} s'est concentré de manière très convaincante sur {zone_desc} "
            f"pour établir le diagnostic de « {disease} ». Ce résultat est cliniquement interprétable."
        )
    elif avg_drop >= 20:
        overall = (
            f"Explicabilité fiable — {valid_count}/{len(drops)} méthodes valides, "
            f"chute moyenne : {avg_drop} %. "
            f"Le modèle {model} a correctement ciblé {zone_desc} pour le diagnostic "
            f"de « {disease} ». Les cartes peuvent être utilisées pour orienter l'examen clinique."
        )
    elif avg_drop >= 10:
        overall = (
            f"Explicabilité partielle — {valid_count}/{len(drops)} méthode(s) valide(s) "
            f"seulement, chute moyenne : {avg_drop} %. "
            f"Le modèle {model} a peut-être considéré des régions non spécifiques à « {disease} ». "
            f"Les cartes doivent être utilisées avec prudence et non comme seul outil de décision."
        )
    else:
        overall = (
            f"Explicabilité insuffisante — aucune méthode n'atteint le seuil de 20 % "
            f"(chute moyenne : {avg_drop} %). "
            f"Les zones activées pour « {disease} » ne correspondent pas clairement aux régions "
            f"pathologiques attendues ({zone_desc}). "
            f"Ce diagnostic nécessite une vérification clinique approfondie, indépendamment de l'IA."
        )

    # Ajout de la confiance et de l'alerte OOD au bilan global
    if conf:
        niv = "élevée" if conf >= 85 else "modérée" if conf >= 70 else "faible"
        overall += f" Confiance du modèle {model} : {conf} % ({niv})."

    if not is_known:
        overall += " ⚠️ Image hors-distribution (OOD) — le modèle n'a pas reconnu de maladie connue."

    return {
        'gradcam':      gradcam_text,
        'shap':         shap_text,
        'hybrid':       hybrid_text,
        'segmentation': segm_text,
        'overall':      overall,
    }


# =============================================================================
#  Ré-entraînement automatique en arrière-plan
# =============================================================================
def _trigger_retraining(doctor_id: int, sample_count: int, app):
    """
    Déclenche le ré-entraînement des deux modèles (EfficientNet + ViT)
    dans un thread daemon pour ne pas bloquer la réponse HTTP.

    Le statut est suivi en temps réel dans la table RetrainingLog :
        pending → running → completed | failed

    Paramètres :
        doctor_id    : ID du médecin qui a déclenché le ré-entraînement
        sample_count : nombre total d'images few-shot utilisées
        app          : instance Flask (nécessaire pour le contexte applicatif)
    """
    # Créer le log initial (pending) dans le contexte de la requête courante
    with app.app_context():
        log = RetrainingLog(
            sample_count = sample_count,
            triggered_by = doctor_id,
            status       = 'pending',
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
                log.notes  = ' ; '.join(errors)
            else:
                log.status = 'completed'

            log.completed_at = datetime.utcnow()
            db.session.commit()

    # Lancer le thread en mode daemon (s'arrête avec le processus principal)
    threading.Thread(target=run, daemon=True).start()
