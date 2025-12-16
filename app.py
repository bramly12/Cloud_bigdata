# API Flask - Middleware entre Streamlit et Databricks
from flask import Flask, jsonify, request, Response
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv
from flask_caching import Cache
from functools import wraps
from datetime import datetime

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)

# Configuration du cache
cache_config = {
    "DEBUG": True,
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 900  # 15 minutes
}
app.config.from_mapping(cache_config)
cache = Cache(app)

# Configuration Databricks
DATABRICKS_HOST = os.getenv("DATABRICKS_INSTANCE")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# =========== Utilitaires pour l'interaction avec Databricks ===========

import time
import json
import requests

def run_databricks_job(job_id, notebook_params, max_wait_time=300, check_interval=0.25):
    """
    Exécute un job Databricks avec des paramètres, attend sa fin et renvoie le résultat.
    
    Args:
        job_id (str): ID du job Databricks à exécuter
        notebook_params (dict): Paramètres à passer au notebook
        max_wait_time (int): Temps d'attente maximum en secondes
        check_interval (int): Intervalle de vérification du statut en secondes
        
    Returns:
        dict or str: Résultat du job Databricks (JSON parsé ou texte brut)
    """
    # Initialisation du timing
    start_total = time.time()
    timings = {}
    
    # Configuration de l'authentification
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Début de la phase de lancement du job
    start_launch = time.time()
    # URL pour lancer le job
    run_job_url = f"{DATABRICKS_HOST}/api/2.0/jobs/run-now"
    payload = {
        "job_id": job_id,
        "notebook_params": notebook_params
    }
    
    # Lancement du job
    response = requests.post(run_job_url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Impossible de lancer le job. Code HTTP: {response.status_code}, "
                        f"Détails: {response.text}")

    run_id = response.json().get("run_id")
    if not run_id:
        raise Exception("Aucun 'run_id' retourné par l'API Databricks.")
    
    # Fin de la phase de lancement
    end_launch = time.time()
    launch_time = end_launch - start_launch
    timings['job_launch'] = launch_time
    print(f"[TIMING] Lancement du job: {launch_time:.2f}s (run_id: {run_id})")

    # Préparation des URL pour le suivi et la récupération du résultat
    status_url = f"{DATABRICKS_HOST}/api/2.0/jobs/runs/get"
    output_url = f"{DATABRICKS_HOST}/api/2.0/jobs/runs/get-output"
    params = {"run_id": run_id}

    # Début de la phase de polling
    start_polling = time.time()
    poll_count = 0
    job_status_times = []
    
    # Boucle d'attente de la fin du job (ou time-out)
    elapsed = 0
    while elapsed < max_wait_time:
        poll_start = time.time()
        status_resp = requests.get(status_url, headers=headers, params=params)
        poll_end = time.time()
        poll_time = poll_end - poll_start
        job_status_times.append(poll_time)
        poll_count += 1
        
        if status_resp.status_code != 200:
            raise Exception(f"Erreur lors de la récupération du statut du job. "
                            f"Code HTTP: {status_resp.status_code}, Détails: {status_resp.text}")
        
        status_data = status_resp.json()
        state = status_data.get("state", {})
        life_cycle_state = state.get("life_cycle_state")
        result_state = state.get("result_state")
        
        # Logging supplémentaire sur l'état du job
        current_state_msg = f"État actuel: {life_cycle_state}"
        if result_state:
            current_state_msg += f", Résultat: {result_state}"
        print(f"[TIMING] Poll #{poll_count}: {poll_time:.2f}s - {current_state_msg}")
        
        # On vérifie si le job est terminé
        if life_cycle_state == "TERMINATED":
            # S'il est terminé avec succès, on sort de la boucle
            if result_state == "SUCCESS":
                break
            # Sinon, on lève une exception pour signaler l'échec
            raise Exception(f"Le job Databricks a échoué. Détails: {state.get('state_message')}")
        
        time.sleep(check_interval)
        elapsed += check_interval

    # Fin de la phase de polling
    end_polling = time.time()
    polling_time = end_polling - start_polling
    timings['job_polling'] = polling_time
    timings['poll_count'] = poll_count
    timings['avg_poll_time'] = sum(job_status_times) / len(job_status_times) if job_status_times else 0
    timings['job_execution_time'] = polling_time - (poll_count * check_interval)
    print(f"[TIMING] Polling total: {polling_time:.2f}s ({poll_count} polls, moyenne: {timings['avg_poll_time']:.2f}s/poll)")
    
    # Vérification du time-out
    if elapsed >= max_wait_time:
        raise TimeoutError(f"Le job n'est pas terminé après {max_wait_time} secondes.")

    # Début de la phase de récupération des résultats
    start_results = time.time()
    # Si on arrive ici, le job est terminé avec succès : on récupère la sortie
    output_resp = requests.get(output_url, headers=headers, params=params)
    if output_resp.status_code != 200:
        raise Exception(f"Erreur lors de la récupération du résultat. "
                        f"Code HTTP: {output_resp.status_code}, Détails: {output_resp.text}")
    
    output_data = output_resp.json()
    
    # Gestion d'une éventuelle erreur renvoyée dans la réponse
    if "error" in output_data:
        raise Exception(f"Une erreur est survenue lors de l'exécution du notebook : {output_data['error']}")
    
    # On retourne la valeur du 'result' dans la section 'notebook_output'
    notebook_output = output_data.get("notebook_output", {})
    result = notebook_output.get("result")
    
    # Fin de la phase de récupération des résultats
    end_results = time.time()
    results_time = end_results - start_results
    timings['results_retrieval'] = results_time
    print(f"[TIMING] Récupération des résultats: {results_time:.2f}s")
    
    # Début de la phase de parsing
    start_parsing = time.time()
    # Tenter de parser le résultat comme JSON, sinon retourner tel quel
    parsed_result = None
    try:
        parsed_result = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        parsed_result = result
    
    # Fin de la phase de parsing
    end_parsing = time.time()
    parsing_time = end_parsing - start_parsing
    timings['json_parsing'] = parsing_time
    print(f"[TIMING] Parsing JSON: {parsing_time:.2f}s")
    
    # Temps total
    end_total = time.time()
    total_time = end_total - start_total
    timings['total_time'] = total_time
    
    # Résumé complet
    print(f"[TIMING] RÉSUMÉ COMPLET pour job_id={job_id}:")
    print(f"  - Temps total: {total_time:.2f}s")
    print(f"  - Lancement du job: {launch_time:.2f}s ({(launch_time/total_time)*100:.1f}%)")
    print(f"  - Polling/attente: {polling_time:.2f}s ({(polling_time/total_time)*100:.1f}%)")
    print(f"  - Temps d'exécution estimé: {timings['job_execution_time']:.2f}s")
    print(f"  - Récupération des résultats: {results_time:.2f}s ({(results_time/total_time)*100:.1f}%)")
    print(f"  - Parsing JSON: {parsing_time:.2f}s ({(parsing_time/total_time)*100:.1f}%)")
    
    # Stocker les métriques pour analyse ultérieure
    try:
        # Par exemple, vous pourriez ajouter ces métriques à un dict global ou une BD
        global execution_metrics
        if 'execution_metrics' not in globals():
            execution_metrics = []
        
        execution_metrics.append({
            'timestamp': time.time(),
            'job_id': job_id,
            'run_id': run_id,
            'timings': timings
        })
    except Exception as e:
        print(f"Erreur lors de l'enregistrement des métriques: {e}")
    
    return parsed_result

def api_error_handler(f):
    """
    Décorateur pour gérer uniformément les erreurs dans les endpoints API
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            app.logger.error(f"Erreur dans l'API: {str(e)}")
            return jsonify({"error": str(e)}), 500
    return decorated_function

# =========== Routes API ===========

@app.route('/get_dataviz', methods=['GET'])
@cache.cached(timeout=600, query_string=True)
@api_error_handler
def get_dataviz():
    """
    Endpoint pour obtenir une visualisation de données.
    Exécute un job Databricks qui génère un graphique Plotly.
    
    Query parameters:
        job_id: ID du job Databricks à exécuter
        analysis_type: Type d'analyse à effectuer
        min_credit: Montant minimum de crédit pour le filtrage
        max_credit: Montant maximum de crédit pour le filtrage
        min_income: Revenu minimum pour le filtrage
    """
    # Récupérer les paramètres
    job_id = request.args.get('job_id', type=str)
    analysis_type = request.args.get('analysis_type', default='2', type=str)
    min_credit = request.args.get('min_credit', default='', type=str)
    max_credit = request.args.get('max_credit', default='', type=str)
    min_income = request.args.get('min_income', default='', type=str)
    
    if not job_id:
        return jsonify({"error": "Le paramètre job_id est requis"}), 400
    
    # Paramètres pour le notebook
    notebook_params = {
        "analysis_type": analysis_type,
        "min_credit": min_credit,
        "max_credit": max_credit,
        "min_income": min_income
    }
    
    # Appel à Databricks avec les paramètres
    data_viz_result = run_databricks_job(job_id, notebook_params)
    
    # Vérifier si le résultat est déjà au format JSON
    if isinstance(data_viz_result, dict):
        analysis_result = data_viz_result
    else:
        # Conversion de la réponse texte en dictionnaire Python
        try:
            analysis_result = json.loads(data_viz_result)
        except json.JSONDecodeError:
            return jsonify({"error": "Format de réponse Databricks invalide"}), 500
    
    # Conversion des données en DataFrame pandas
    df = pd.DataFrame(analysis_result['data'])
    
    # Récupération des métadonnées
    metadata = analysis_result['metadata']
    
    # Création du graphique Plotly en fonction du type de graphique
    fig = create_plotly_figure(df, metadata)
    
    # Conversion du graphique en HTML
    html_content = fig.to_html(include_plotlyjs=True, full_html=True)
    
    # Retourner le HTML directement
    return Response(html_content, mimetype='text/html')

# Amélioration de la fonction predict_default dans app.py
@app.route('/predict_default', methods=['GET'])
@cache.cached(timeout=600, query_string=True)
@api_error_handler
def predict_default():
    """
    Endpoint pour prédire le risque de défaut d'un client.
    Version améliorée avec un formatage des données plus professionnel.
    """
    # Récupérer les paramètres
    job_id = request.args.get('job_id', type=str)
    client_id = request.args.get('client_id', type=str)
    
    if not job_id:
        return jsonify({"error": "Le paramètre job_id est requis"}), 400
    
    if not client_id:
        return jsonify({"error": "Le paramètre client_id est requis"}), 400
    
    # Paramètres pour le notebook
    notebook_params = {
        "client_id": client_id
    }
    
    try:
        # Appel à Databricks avec les paramètres
        t0 = time.time()
        prediction_result = run_databricks_job(job_id, notebook_params)
        print("Databricks : ", time.time()-t0)
        
        # Si le résultat n'est pas un dictionnaire (ce qui devrait être le cas),
        # c'est probablement une erreur
        if not isinstance(prediction_result, dict):
            return jsonify({"error": "Format de réponse Databricks invalide"}), 500
        
        # En cas d'erreur dans la prédiction
        if "error" in prediction_result:
            return jsonify({"error": prediction_result["error"]}), 500
        
        # Formater les données client pour un affichage plus professionnel
        if "client_info" in prediction_result:
            prediction_result["client_info"] = format_client_data(prediction_result["client_info"])
        
        # Ajouter des recommandations basées sur le score de risque
        if "prediction" in prediction_result and "risk_score" in prediction_result["prediction"]:
            risk_score = prediction_result["prediction"]["risk_score"]
            prediction_result["recommendation"] = get_recommendation(risk_score)
            
            # Ajouter des métadonnées supplémentaires pour l'interface
            prediction_result["metadata"] = {
                "analysis_date": datetime.now().strftime("%d/%m/%Y"),
                "analysis_time": datetime.now().strftime("%H:%M"),
                "version": "1.2.3",
                "model_type": "Régression logistique",
                "data_source": "Historique clients 2020-2024"
            }
        
        return jsonify(prediction_result)
        
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Impossible de se connecter à Databricks. Vérifiez la configuration."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "La requête a expiré. Le service Databricks met trop de temps à répondre."}), 504
    except Exception as e:
        return jsonify({"error": f"Erreur inattendue: {str(e)}"}), 500

# Validation supplémentaire pour le format des nombres
def validate_number_format(value):
    """
    Vérifie et corrige le format des nombres pour éviter les problèmes d'affichage
    comme les décimales excessives.
    
    Args:
        value: La valeur à valider/corriger
        
    Returns:
        La valeur corrigée si nécessaire
    """
    # Si c'est un nombre flottant avec beaucoup de décimales
    if isinstance(value, float):
        # Pour les nombres entiers représentés comme flottants (ex: 42.0)
        if value.is_integer():
            return int(value)
        # Pour les années d'emploi ou autres métriques qui devraient avoir peu de décimales
        if abs(value) < 100:
            return round(value, 1)
        # Pour les grands nombres comme les montants financiers
        return round(value)
    return value

# Helper pour normaliser les facteurs d'influence
def normalize_impact_factors(impact_factors):
    """
    Normalise les facteurs d'influence pour assurer une présentation cohérente.
    
    Args:
        impact_factors (list): Liste de facteurs d'influence
        
    Returns:
        list: Liste normalisée de facteurs d'influence
    """
    # Trouver l'impact maximum en valeur absolue pour normaliser
    max_abs_impact = max([abs(factor["impact"]) for factor in impact_factors]) if impact_factors else 1
    
    # Normaliser chaque facteur
    for factor in impact_factors:
        # Normaliser l'impact en pourcentage du maximum (pour visualisation)
        factor["impact_normalized"] = round(100 * abs(factor["impact"]) / max_abs_impact)
        
        # Assurer que les valeurs sont bien formatées
        if "value" in factor and isinstance(factor["value"], float):
            factor["value"] = validate_number_format(factor["value"])
            
        # Ajouter une couleur pour l'UI
        if factor["impact"] > 0:
            factor["color"] = "#e74c3c"  # Rouge pour les facteurs négatifs
        else:
            factor["color"] = "#4CAF50"  # Vert pour les facteurs positifs
    
    return impact_factors

# Exemple d'utilisation dans la route API
@app.route('/get_impact_factors', methods=['GET'])
@cache.cached(timeout=600, query_string=True)
@api_error_handler
def get_impact_factors():
    """
    Endpoint pour récupérer les facteurs d'influence formatés pour un client spécifique.
    """
    client_id = request.args.get('client_id', type=str)
    job_id = request.args.get('job_id', type=str)
    
    if not client_id or not job_id:
        return jsonify({"error": "Les paramètres client_id et job_id sont requis"}), 400
    
    # Appeler le même endpoint de prédiction
    prediction_result = run_databricks_job(job_id, {"client_id": client_id})
    
    # Extraire et formater les facteurs d'influence
    if isinstance(prediction_result, dict) and "impact_factors" in prediction_result:
        impact_factors = prediction_result["impact_factors"]
        normalized_factors = normalize_impact_factors(impact_factors)
        
        return jsonify({
            "client_id": client_id,
            "impact_factors": normalized_factors,
            "metadata": {
                "analysis_date": datetime.now().strftime("%d/%m/%Y"),
                "factor_count": len(normalized_factors)
            }
        })
    else:
        return jsonify({"error": "Impossible d'extraire les facteurs d'influence"}), 500

# Améliorations à apporter au modèle de données de l'API Flask

def get_recommendation(risk_score):
    """
    Renvoie une recommandation en fonction du score de risque.
    Version améliorée avec formatage plus professionnel.
    
    Args:
        risk_score (float): Score entre 0 et 1 indiquant la probabilité de défaut.
        
    Returns:
        dict: Dictionnaire contenant la recommandation et sa justification.
    """
    if risk_score >= 0.70:
        return {
            "decision": "Refus recommandé",
            "explanation": "Le niveau de risque est trop élevé pour accorder le crédit aux conditions demandées.",
            "alternative": "Proposer un montant réduit avec garanties supplémentaires.",
            "risk_level": "Élevé",
            "action_plan": [
                "Refuser le crédit aux conditions demandées",
                "Proposer un montant réduit (max 30% du revenu)",
                "Exiger des garanties supplémentaires (caution, hypothèque)",
                "Orienter le client vers un accompagnement budgétaire"
            ]
        }
    elif risk_score >= 0.40:
        return {
            "decision": "Acceptation conditionnelle",
            "explanation": "Le dossier présente un risque modéré qui nécessite des garanties supplémentaires.",
            "alternative": "Envisager un co-emprunteur ou une caution solidaire.",
            "risk_level": "Moyen",
            "action_plan": [
                "Accepter sous réserve de garanties supplémentaires",
                "Ajuster le taux d'intérêt pour compenser le risque",
                "Proposer une durée d'emprunt plus courte",
                "Mettre en place un suivi trimestriel"
            ]
        }
    else:
        return {
            "decision": "Acceptation recommandée",
            "explanation": "Le profil présente un faible risque de défaut.",
            "alternative": "Procéder à l'octroi du crédit aux conditions standards.",
            "risk_level": "Faible",
            "action_plan": [
                "Accorder le crédit aux conditions standards",
                "Proposer des produits complémentaires (assurance, épargne)",
                "Appliquer le processus de suivi standard",
                "Réévaluer le risque annuellement"
            ]
        }

# Fonction pour formater les données client retournées par Databricks
def format_client_data(client_info):
    """
    Formate les données client pour un affichage plus professionnel.
    Corrige les problèmes de formatage des nombres et dates.
    
    Args:
        client_info (dict): Informations brutes du client
        
    Returns:
        dict: Informations client formatées
    """
    formatted_info = client_info.copy()
    
    # Formatage de l'âge
    if "age" in formatted_info and isinstance(formatted_info["age"], (int, float)):
        formatted_info["age"] = int(formatted_info["age"])
        
    # Formatage de l'ancienneté d'emploi
    if "years_employed" in formatted_info and isinstance(formatted_info["years_employed"], float):
        # Limiter les décimales excessives (comme vu dans l'exemple avec 3.258...)
        if formatted_info["years_employed"] > 100:  # Une valeur anormalement grande
            formatted_info["years_employed"] = round(formatted_info["years_employed"] % 100, 1)
        else:
            formatted_info["years_employed"] = round(formatted_info["years_employed"], 1)
    
    # Formatage des montants financiers
    for field in ["income", "credit_amount"]:
        if field in formatted_info and formatted_info[field] is not None:
            # Arrondir à l'entier le plus proche
            formatted_info[field] = round(float(formatted_info[field]))
    
    # Traduction des valeurs en français si nécessaire
    translations = {
        "gender": {"M": "Homme", "F": "Femme"},
        "income_type": {
            "Working": "Salarié", 
            "Commercial associate": "Commercial",
            "Pensioner": "Retraité",
            "State servant": "Fonctionnaire",
            "Entrepreneur": "Entrepreneur"
        },
        "education_type": {
            "Higher education": "Enseignement supérieur",
            "Secondary / secondary special": "Secondaire",
            "Incomplete higher": "Supérieur incomplet",
            "Lower secondary": "Premier cycle secondaire",
            "Academic degree": "Diplôme universitaire"
        },
        "family_status": {
            "Married": "Marié(e)",
            "Single / not married": "Célibataire",
            "Civil marriage": "Pacs",
            "Separated": "Séparé(e)",
            "Widow": "Veuf/veuve"
        },
        "housing_type": {
            "House / apartment": "Maison / appartement",
            "Rented apartment": "Location",
            "With parents": "Chez les parents",
            "Municipal apartment": "Logement social",
            "Office apartment": "Logement de fonction",
            "Co-op apartment": "Coopérative"
        }
    }
    
    # Appliquer les traductions
    for field, translation_dict in translations.items():
        if field in formatted_info and formatted_info[field] in translation_dict:
            formatted_info[field] = translation_dict[formatted_info[field]]
    
    return formatted_info

def create_plotly_figure(df, metadata):
    """Crée un graphique Plotly en fonction des métadonnées et du DataFrame"""
    
    chart_type = metadata['chart_type']
    title = metadata['title']
    
    if chart_type == "bar":
        x_col = metadata['x_col']
        y_col = metadata['y_col']
        fig = px.bar(df, x=x_col, y=y_col, color="CLIENT_COUNT",
                    title=title,
                    labels={y_col: "Taux de défaut (%)", x_col: x_col.replace("_", " ")},
                    color_continuous_scale=px.colors.sequential.Viridis)
    
    elif chart_type == "scatter":
        x_col = metadata['x_col']
        y_col = metadata['y_col']
        
        # Vérifier si 'COUNT' ou 'CLIENT_COUNT' est présent pour la taille des points
        size_col = "COUNT" if "COUNT" in df.columns else "CLIENT_COUNT"
        
        fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=y_col,
                       title=title,
                       labels={x_col: x_col.replace("_", " "), y_col: "Taux de défaut (%)"},
                       color_continuous_scale=px.colors.sequential.Plasma)
        
        fig.update_traces(mode='markers+lines')
    
    elif chart_type == "bar_grouped":
        x_col = metadata['x_col']
        y_col = metadata['y_col']
        group_col = metadata['group_col']
        
        fig = px.bar(df, x=x_col, y=y_col, color=group_col, barmode="group",
                   title=title,
                   labels={x_col: x_col.replace("_", " "), y_col: "Taux de défaut (%)"},
                   color_discrete_sequence=px.colors.qualitative.Set1)
        
        fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    
    elif chart_type == "heatmap":
        x_col = metadata['x_col']
        y_col = metadata['y_col']
        z_col = metadata['z_col']
        
        fig = px.density_heatmap(df, x=x_col, y=y_col, z=z_col,
                              color_continuous_scale=px.colors.sequential.Plasma,
                              title=title,
                              labels={x_col: x_col.replace("_", " "), 
                                     y_col: y_col.replace("_", " "),
                                     z_col: "Taux de défaut (%)"},
                              text_auto=True)
        
        fig.update_traces(texttemplate='%{z:.2f}%')
    
    elif chart_type == "polar":
        theta_col = metadata['theta_col']
        r_col = metadata['r_col']
        
        fig = px.line_polar(df, r=r_col, theta=theta_col, line_close=True,
                         color_discrete_sequence=["red"],
                         title=title)
        fig.update_traces(fill='toself')
    
    else:
        # Type de graphique non reconnu, on crée un graphique vide avec un message d'erreur
        fig = go.Figure()
        fig.add_annotation(text=f"Type de graphique non reconnu: {chart_type}",
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
    
    # Améliorer la mise en page pour tous les graphiques
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        template='plotly_white',
        height=700
    )
    
    return fig

if __name__ == "__main__":
    # Pour le développement local
    app.run(debug=True, host='0.0.0.0', port=5001)