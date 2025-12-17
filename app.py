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
from typing import Dict, Any, Optional, Union
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging


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
CLUSTER_ID = os.getenv("CLUSTER_ID")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# Setup logger
logger = logging.getLogger('simple_logger')
logger.setLevel(logging.INFO)
logger.addHandler(AzureLogHandler(connection_string=CONNECTION_STRING))

@app.route('/feedback', methods=['POST'])
def log_message():
    """
    POST /log
    Body: {
        "message": "Texte à logger",
        "custom_dimensions": {"key": "value"},
        "is_positive": true/false
    }
    """
    data = request.get_json()
    
    message = data.get('message', 'Message vide')
    custom_dimensions = data.get('custom_dimensions', {})
    is_positive = data.get('is_positive', True)
    
    if is_positive:
        logger.info(message, extra={'custom_dimensions': custom_dimensions})
    else:
        logger.error(message, extra={'custom_dimensions': custom_dimensions})
    
    print("Logs envoyés à Applications Insights !")
    
    return jsonify({'success': True, 'logged': message})

# =========== Utilitaires pour l'interaction avec Databricks ===========

def create_databricks_context() -> str:
    """
    Crée un contexte d'exécution sur le cluster Databricks et exécute une
    commande très simple pour vérifier son bon fonctionnement.

    Returns:
        str: L'ID du contexte créé

    Raises:
        Exception: Si la création du contexte échoue
    """
    try:
        # En-têtes pour toutes les requêtes
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }

        # 1. Créer un contexte d'exécution
        create_context_url = f"{DATABRICKS_HOST}/api/1.2/contexts/create"
        create_context_payload = {
            "clusterId": CLUSTER_ID,
            "language": "python"
        }

        response = requests.post(create_context_url, headers=headers, json=create_context_payload)
        response.raise_for_status()

        context_id = response.json().get("id")
        print(f"Contexte créé avec ID: {context_id}")

        return context_id

    except Exception as e:
        raise Exception(f"Erreur lors de la création du contexte: {str(e)}")

# Créer un contexte d'exécution
CONTEXT_ID = create_databricks_context()

def check_context_validity(context_id: str) -> bool:
    """
    Vérifie si un contexte d'exécution Databricks est toujours valide.

    Args:
        context_id: L'ID du contexte à vérifier

    Returns:
        bool: True si le contexte est valide, False sinon
    """
    try:
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }

        # Exécuter une commande simple pour vérifier si le contexte est valide
        execute_url = f"{DATABRICKS_HOST}/api/1.2/commands/execute"
        execute_payload = {
            "clusterId": CLUSTER_ID,
            "contextId": context_id,
            "language": "python",
            "command": "print('Contexte valide')"
        }

        response = requests.post(execute_url, headers=headers, json=execute_payload)

        # Si la requête échoue, le contexte n'est probablement plus valide
        if response.status_code != 200:
            return False

        command_id = response.json().get("id")

        # Vérifier que la commande s'exécute correctement
        status_url = f"{DATABRICKS_HOST}/api/1.2/commands/status"
        status_params = {
            "clusterId": CLUSTER_ID,
            "contextId": context_id,
            "commandId": command_id
        }

        for _ in range(5):  # Quelques tentatives
            response = requests.get(status_url, headers=headers, params=status_params)
            if response.status_code != 200:
                return False

            status_data = response.json()
            status = status_data.get("status")

            if status == "Finished":
                return True
            elif status == "Error":
                return False

            time.sleep(0.5)

        return False  # Timeout

    except Exception:
        return False  # En cas d'erreur, considérer que le contexte n'est pas valide

def get_dataviz_from_databricks(analysis_type: int, min_credit: Optional[float] = None,
                max_credit: Optional[float] = None, min_income: Optional[float] = None, context_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Récupère une visualisation de données depuis Databricks en exécutant une analyse directement
    sur le cluster via l'API REST.

    Args:
        analysis_type: Type d'analyse à effectuer (1-8)
        min_credit: Montant minimum de crédit pour le filtrage (optionnel)
        max_credit: Montant maximum de crédit pour le filtrage (optionnel)
        min_income: Revenu minimum pour le filtrage (optionnel)

    Returns:
        Dict: Résultats de l'analyse avec métadonnées pour visualisation
    """
    try:
        created_new_context = False
        # En-têtes pour toutes les requêtes
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }

        # Utiliser le contexte fourni ou en créer un nouveau
        if context_id is not None:
            # Vérifier que le contexte est toujours valide
            if not check_context_validity(context_id):
                print(f"Contexte {context_id} invalide, création d'un nouveau contexte...")
                context_id = create_databricks_context()
                created_new_context = True
            else:
                print(f"Réutilisation du contexte: {context_id}")
        else:
            # Créer un nouveau contexte
            context_id = create_databricks_context()
            created_new_context = True

        # 2. Construire les filtres
        filters_str = "{"
        if min_credit is not None:
            filters_str += f"'min_credit': {min_credit}, "
        if max_credit is not None:
            filters_str += f"'max_credit': {max_credit}, "
        if min_income is not None:
            filters_str += f"'min_income': {min_income}, "
        filters_str = filters_str.rstrip(", ") + "}"

        # 3. Code Python à exécuter pour l'analyse
        command = f"""
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, when, round, ntile, avg, count, coalesce, lit
        from pyspark.sql.window import Window
        import pandas as pd
        import json

        # Fonction pour récupérer le DataFrame mis en cache ou le charger si nécessaire
        def get_cached_df():
            # Vérifier si le DataFrame est déjà dans le cache global
            if 'cached_df' not in globals():
                #print("[INFO] Chargement du DataFrame depuis DBFS...")
                dbfs_path = "dbfs:/application_train.csv"

                # Lecture du fichier CSV
                df = spark.read.format("csv") \\
                    .option("header", "true") \\
                    .option("inferSchema", "true") \\
                    .load(dbfs_path)

                # Mise en cache du DataFrame pour accélérer les opérations ultérieures
                df = df.cache()

                # Forcer l'évaluation du cache
                count = df.count()
                # print(f"[INFO] DataFrame chargé et mis en cache avec {{count}} lignes")

                # Stocker dans le cache global
                globals()['cached_df'] = df
            else:
                pass
                #print("[INFO] Utilisation du DataFrame depuis le cache")

            return globals()['cached_df']

        def process_credit_analysis(df, analysis_type=1, filters=None):
            #print(f"[INFO] Exécution de l'analyse de type {{analysis_type}} avec les filtres {{filters}}")

            # Appliquer les filtres si présents
            filtered_df = df
            if filters:
                if 'min_credit' in filters and filters['min_credit']:
                    filtered_df = filtered_df.filter(col("AMT_CREDIT") >= filters['min_credit'])
                if 'max_credit' in filters and filters['max_credit']:
                    filtered_df = filtered_df.filter(col("AMT_CREDIT") <= filters['max_credit'])
                if 'min_income' in filters and filters['min_income']:
                    filtered_df = filtered_df.filter(col("AMT_INCOME_TOTAL") >= filters['min_income'])

            # Exécuter l'analyse correspondante
            if analysis_type == 1:
                # Analyse de l'âge des clients
                filtered_df = filtered_df.withColumn("AGE", -col("DAYS_BIRTH") / 365.25)
                filtered_df = filtered_df.withColumn("AGE_GROUP",
                                   when(col("AGE") < 25, "< 25 ans")
                                   .when(col("AGE") < 35, "25-35 ans")
                                   .when(col("AGE") < 45, "35-45 ans")
                                   .when(col("AGE") < 55, "45-55 ans")
                                   .otherwise("> 55 ans"))

                result_df = filtered_df.groupBy("AGE_GROUP").agg(
                    count("*").alias("CLIENT_COUNT"),
                    round(avg(when(col("TARGET") == 1, 1).otherwise(0)) * 100, 2).alias("DEFAULT_RATE")
                ).toPandas()

                chart_type = "bar"
                title = "Taux de défaut par groupe age"
                x_col = "AGE_GROUP"
                y_col = "DEFAULT_RATE"

            elif analysis_type == 2:
                # Analyse du ratio crédit/revenu
                filtered_df = filtered_df.withColumn("CREDIT_INCOME_RATIO", round(col("AMT_CREDIT") / col("AMT_INCOME_TOTAL"), 2))
                filtered_df = filtered_df.withColumn("RATIO_CATEGORY", ntile(5).over(Window.orderBy("CREDIT_INCOME_RATIO")))

                result_df = filtered_df.groupBy("RATIO_CATEGORY").agg(
                    count("*").alias("COUNT"),
                    round(avg("CREDIT_INCOME_RATIO"), 2).alias("AVG_RATIO"),
                    round(avg(when(col("TARGET") == 1, 1).otherwise(0)) * 100, 2).alias("DEFAULT_RATE")
                ).toPandas()

                chart_type = "scatter"
                title = "Relation entre ratio crédit/revenu et taux de défaut"
                x_col = "AVG_RATIO"
                y_col = "DEFAULT_RATE"

            elif analysis_type == 3:
                # Analyse de l'impact du type de revenu
                result_df = filtered_df.groupBy("NAME_INCOME_TYPE").agg(
                    count("*").alias("CLIENT_COUNT"),
                    round(avg("AMT_INCOME_TOTAL"), 0).alias("AVG_INCOME"),
                    round(avg(when(col("TARGET") == 1, 1).otherwise(0)) * 100, 2).alias("DEFAULT_RATE")
                ).filter(col("CLIENT_COUNT") > 100).toPandas()

                chart_type = "scatter"
                title = "Type de revenu, montant et taux de défaut"
                x_col = "AVG_INCOME"
                y_col = "DEFAULT_RATE"

            elif analysis_type == 4:
                # Analyse ancienneté emploi
                filtered_df = filtered_df.withColumn("YEARS_EMPLOYED",
                                  when(col("DAYS_EMPLOYED") == 365243, 0)  # Valeur pour les sans emploi
                                  .otherwise(-col("DAYS_EMPLOYED") / 365.25))
                filtered_df = filtered_df.withColumn("EMPLOYMENT_LENGTH",
                                  when(col("YEARS_EMPLOYED") < 1, "< 1 an")
                                  .when(col("YEARS_EMPLOYED") < 3, "1-3 ans")
                                  .when(col("YEARS_EMPLOYED") < 5, "3-5 ans")
                                  .when(col("YEARS_EMPLOYED") < 10, "5-10 ans")
                                  .otherwise("> 10 ans"))

                result_df = filtered_df.groupBy("EMPLOYMENT_LENGTH").agg(
                    count("*").alias("CLIENT_COUNT"),
                    round(avg(when(col("TARGET") == 1, 1).otherwise(0)) * 100, 2).alias("DEFAULT_RATE")
                ).orderBy("EMPLOYMENT_LENGTH").toPandas()

                chart_type = "bar"
                title = "Impact ancienneté emploi sur le taux de défaut"
                x_col = "EMPLOYMENT_LENGTH"
                y_col = "DEFAULT_RATE"

            elif analysis_type == 5:
                # Analyse des demandes de bureau de crédit
                filtered_df = filtered_df.withColumn("TOTAL_ENQUIRIES",
                                  coalesce(col("AMT_REQ_CREDIT_BUREAU_HOUR"), lit(0)) +
                                  coalesce(col("AMT_REQ_CREDIT_BUREAU_DAY"), lit(0)) +
                                  coalesce(col("AMT_REQ_CREDIT_BUREAU_WEEK"), lit(0)) +
                                  coalesce(col("AMT_REQ_CREDIT_BUREAU_MON"), lit(0)) +
                                  coalesce(col("AMT_REQ_CREDIT_BUREAU_QRT"), lit(0)) +
                                  coalesce(col("AMT_REQ_CREDIT_BUREAU_YEAR"), lit(0)))

                filtered_df = filtered_df.withColumn("ENQUIRY_GROUP",
                                  when(col("TOTAL_ENQUIRIES") == 0, "Aucune demande")
                                  .when(col("TOTAL_ENQUIRIES") <= 2, "1-2 demandes")
                                  .when(col("TOTAL_ENQUIRIES") <= 5, "3-5 demandes")
                                  .when(col("TOTAL_ENQUIRIES") <= 10, "6-10 demandes")
                                  .otherwise("> 10 demandes"))

                result_df = filtered_df.groupBy("ENQUIRY_GROUP").agg(
                    count("*").alias("CLIENT_COUNT"),
                    round(avg(when(col("TARGET") == 1, 1).otherwise(0)) * 100, 2).alias("DEFAULT_RATE")
                ).toPandas()

                chart_type = "scatter"
                title = "Impact des demandes de crédit sur le taux de défaut"
                x_col = "ENQUIRY_GROUP"
                y_col = "DEFAULT_RATE"

            elif analysis_type == 6:
                # Analyse multivariée: Type de contrat, bien immobilier
                result_df = filtered_df.groupBy("NAME_CONTRACT_TYPE", "FLAG_OWN_REALTY").agg(
                    count("*").alias("CLIENT_COUNT"),
                    round(avg("AMT_CREDIT"), 0).alias("AVG_CREDIT"),
                    round(avg(when(col("TARGET") == 1, 1).otherwise(0)) * 100, 2).alias("DEFAULT_RATE")
                ).toPandas()

                result_df["FLAG_OWN_REALTY"] = result_df["FLAG_OWN_REALTY"].map({{
                    "Y": "Propriétaire", "N": "Non propriétaire"
                }})

                chart_type = "bar_grouped"
                title = "Taux de défaut par type de contrat et propriété immobilière"
                x_col = "NAME_CONTRACT_TYPE"
                y_col = "DEFAULT_RATE"
                group_col = "FLAG_OWN_REALTY"

            elif analysis_type == 7:
                # Impact combiné du nombre d'enfants et du statut familial
                result_df = filtered_df.groupBy("NAME_FAMILY_STATUS", "CNT_CHILDREN").agg(
                    count("*").alias("CLIENT_COUNT"),
                    round(avg(when(col("TARGET") == 1, 1).otherwise(0)) * 100, 2).alias("DEFAULT_RATE")
                ).filter(col("CLIENT_COUNT") > 50).toPandas()

                chart_type = "heatmap"
                title = "Taux de défaut selon le statut familial et le nombre enfants"
                x_col = "NAME_FAMILY_STATUS"
                y_col = "CNT_CHILDREN"
                z_col = "DEFAULT_RATE"

            elif analysis_type == 8:
                # Analyse saisonnière selon le jour de la semaine
                result_df = filtered_df.groupBy("WEEKDAY_APPR_PROCESS_START").agg(
                    count("*").alias("APPLICATION_COUNT"),
                    round(avg(when(col("TARGET") == 1, 1).otherwise(0)) * 100, 2).alias("DEFAULT_RATE"),
                    round(avg("AMT_CREDIT"), 0).alias("AVG_CREDIT")
                ).toPandas()

                # Convertir les jours en ordre correct
                weekday_order = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
                result_df["WEEKDAY_ORDER"] = result_df["WEEKDAY_APPR_PROCESS_START"].apply(
                    lambda x: weekday_order.index(x.upper()) if isinstance(x, str) else 0
                )
                result_df = result_df.sort_values("WEEKDAY_ORDER")

                chart_type = "polar"
                title = "Variation du taux de défaut selon le jour de la semaine"
                theta_col = "WEEKDAY_APPR_PROCESS_START"
                r_col = "DEFAULT_RATE"

            #print(f"[INFO] Analyse terminée, préparation des résultats")

            # Retourner les résultats sous forme de dictionnaire JSON
            result = {{
                "data": result_df.to_dict(orient="records"),
                "metadata": {{
                    "analysis_type": analysis_type,
                    "chart_type": chart_type,
                    "title": title,
                    "filters": filters
                }}
            }}

            # Ajouter des métadonnées spécifiques à chaque type de graphique
            if chart_type in ["bar", "scatter", "line"]:
                result["metadata"]["x_col"] = x_col
                result["metadata"]["y_col"] = y_col
            elif chart_type == "bar_grouped":
                result["metadata"]["x_col"] = x_col
                result["metadata"]["y_col"] = y_col
                result["metadata"]["group_col"] = group_col
            elif chart_type == "heatmap":
                result["metadata"]["x_col"] = x_col
                result["metadata"]["y_col"] = y_col
                result["metadata"]["z_col"] = z_col
            elif chart_type == "polar":
                result["metadata"]["theta_col"] = theta_col
                result["metadata"]["r_col"] = r_col

            return result

        # Obtenir le DataFrame à partir du cache
        df = get_cached_df()

        # Exécuter l'analyse et retourner les résultats
        analysis_result = process_credit_analysis(df, {analysis_type}, {filters_str})
        analysis_result
        """

        # 4. Exécuter la commande
        execute_url = f"{DATABRICKS_HOST}/api/1.2/commands/execute"
        execute_payload = {
            "clusterId": CLUSTER_ID,
            "contextId": context_id,
            "language": "python",
            "command": command
        }
        response = requests.post(execute_url, headers=headers, json=execute_payload)

        command_id = response.json().get("id")
        print(f"Commande exécutée avec ID: {command_id}")

        # 5. Attendre et obtenir le résultat
        status_url = f"{DATABRICKS_HOST}/api/1.2/commands/status"
        status_params = {
            "clusterId": CLUSTER_ID,
            "contextId": context_id,
            "commandId": command_id
        }

        max_retries = 30
        retry_count = 0

        while retry_count < max_retries:
            response = requests.get(status_url, headers=headers, params=status_params)

            status_data = response.json()
            status = status_data.get("status")

            if status == "Finished":
                # Commande terminée, extraire le résultat
                result_data = status_data.get("results", {}).get("data").replace("\n", "").replace("'", '"')
                print("==========================",result_data)
                print("==========================",type(result_data))

                if result_data:
                    try:
                        return json.loads(result_data)
                    except json.JSONDecodeError:
                        return {"error": f"Impossible de parser le résultat: {result_data[:100]}..."}
                else:
                    return {"error": "Aucun résultat retourné"}

            elif status == "Error":
                error_message = status_data.get("results", {}).get("summary", "Erreur inconnue")
                return {"error": f"Erreur d'exécution: {error_message}"}

            # Attendre avant de vérifier à nouveau
            time.sleep(1)
            retry_count += 1

        return {{"error": "Délai d'attente dépassé"}}

    except Exception as e:
        import traceback
        return {"error": f"Exception: {str(e)}\n{traceback.format_exc()}"}

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

def predict_default_from_databricks(client_id: str, context_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Prédit le risque de défaut pour un client spécifique en exécutant le code directement
    sur le cluster Databricks via l'API REST.

    Args:
        client_id: ID du client pour lequel effectuer la prédiction
        context_id: ID du contexte Databricks existant (optionnel)

    Returns:
        Dict: Résultat de la prédiction avec les informations du client et le score de risque
    """
    try:
        created_new_context = False
        # En-têtes pour toutes les requêtes
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }

        # Utiliser le contexte fourni ou en créer un nouveau
        if context_id is not None:
            # Vérifier que le contexte est toujours valide
            if not check_context_validity(context_id):
                print(f"Contexte {context_id} invalide, création d'un nouveau contexte...")
                context_id = create_databricks_context()
                created_new_context = True
            else:
                print(f"Réutilisation du contexte: {context_id}")
        else:
            # Créer un nouveau contexte
            context_id = create_databricks_context()
            created_new_context = True

        # Code Python à exécuter pour la prédiction
        command = """
        from pyspark.sql import SparkSession
        import pyspark.sql.functions as F
        from pyspark.ml import PipelineModel
        from pyspark.ml.classification import LogisticRegressionModel
        from pyspark.sql.types import DoubleType
        import json

        # Listes de colonnes à utiliser
        numeric_cols = [
            "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
            "AMT_GOODS_PRICE", "REGION_POPULATION_RELATIVE", "DAYS_BIRTH",
            "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
            "OWN_CAR_AGE", "CNT_FAM_MEMBERS", "EXT_SOURCE_1",
            "EXT_SOURCE_2", "EXT_SOURCE_3", "OBS_30_CNT_SOCIAL_CIRCLE",
            "DEF_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE",
            "DEF_60_CNT_SOCIAL_CIRCLE", "DAYS_LAST_PHONE_CHANGE",
        ]

        categorical_cols = [
            "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_TYPE_SUITE",
            "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "WEEKDAY_APPR_PROCESS_START",
            "ORGANIZATION_TYPE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE",
            "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
        ]

        flag_cols = [
            "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
        ]

        # Chemins des modèles et données
        PREPROCESSOR_PATH = "dbfs:/preprocessor.spark"
        MODEL_PATH = "dbfs:/model.spark"
        CSV_PATH = "dbfs:/test.csv"

        # Définir les fonctions de récupération des ressources mises en cache
        def get_preprocessed_df():
            # Récupère le DataFrame prétraité depuis le cache s'il existe,
            # sinon le charge depuis DBFS et applique les prétraitements
            if 'cached_df' not in globals():
                # print("[INFO] Chargement des données depuis DBFS...")

                # Charger le DataFrame
                df = spark.read.option("header", "true").option("inferSchema", "true").csv(CSV_PATH)

                # Prétraiter le DataFrame
                # print("[INFO] Application des prétraitements au DataFrame...")

                # CAST des colonnes numériques
                for col in numeric_cols:
                    if col in df.columns:
                        df = df.withColumn(col, F.col(col).cast("double"))

                # CAST et imputation des colonnes catégorielles
                for col in categorical_cols:
                    if col in df.columns:
                        df = df.withColumn(col, F.col(col).cast("string"))
                        df = df.withColumn(col, F.coalesce(F.col(col), F.lit("Unknown")))

                # Encodage des colonnes binaires
                for col in flag_cols:
                    if col in df.columns:
                        df = df.withColumn(col, F.when(F.col(col).isin("Y", "1"), 1).otherwise(0))

                # Mettre en cache et forcer l'évaluation
                df = df.cache()
                count = df.count()

                # Stocker dans le cache global
                globals()['cached_df'] = df

                # print(f"[INFO] DataFrame chargé et prétraité avec {{count}} lignes")
            else:
                pass
                # print("[INFO] Utilisation du DataFrame depuis le cache")

            return globals()['cached_df']

        def get_preprocessor():
            # Récupère le preprocessor depuis le cache s'il existe,
            # sinon le charge depuis DBFS
            if 'cached_preprocessor' not in globals():
                # print(f"[INFO] Chargement du preprocessor depuis {{PREPROCESSOR_PATH}}")

                # Charger le preprocessor
                preprocessor = PipelineModel.load(PREPROCESSOR_PATH)

                # Stocker dans le cache global
                globals()['cached_preprocessor'] = preprocessor

                # print("[INFO] Preprocessor chargé")
            else:
                pass
                # print("[INFO] Utilisation du preprocessor depuis le cache")

            return globals()['cached_preprocessor']

        def get_model():
            # Récupère le modèle depuis le cache s'il existe,
            # sinon le charge depuis DBFS
            if 'cached_model' not in globals():
                # print(f"[INFO] Chargement du modèle depuis {{MODEL_PATH}}")

                # Charger le modèle
                model = LogisticRegressionModel.load(MODEL_PATH)

                # Stocker dans le cache global
                globals()['cached_model'] = model

                # print("[INFO] Modèle chargé")
            else:
                pass
                # print("[INFO] Utilisation du modèle depuis le cache")

            return globals()['cached_model']

        # Fonction de prédiction principale
        def predict_for_client(client_id):
            try:
                # 1. Récupérer le DataFrame prétraité
                df = get_preprocessed_df()

                # 2. Filtrer pour le client spécifique
                client_df = df.filter(F.col("SK_ID_CURR") == client_id)

                # Vérifier si le client existe
                client_count = client_df.count()
                if client_count == 0:
                    return {{"error": f"Client ID {{client_id}} non trouvé dans les données"}}

                # 3. Récupérer et appliquer le preprocessor
                preprocessor = get_preprocessor()
                client_df_prepared = preprocessor.transform(client_df)

                # 4. Récupérer et appliquer le modèle
                model = get_model()
                prediction_df = model.transform(client_df_prepared)

                # 5. Extraire le score de risque
                @F.udf(returnType=DoubleType())
                def extract_probability(prob_vector):
                    return float(prob_vector[1])

                try:
                    prediction_df = prediction_df.withColumn("risk_score", extract_probability(F.col("probability")))
                except Exception as e:
                    print(f"[WARN] Impossible d'extraire probability: {{str(e)}}")
                    prediction_df = prediction_df.withColumn(
                        "risk_score",
                        F.when(F.col("prediction") > 0, 0.75).otherwise(0.25)
                    )

                # 6. Extraire les informations du client
                prediction_row = prediction_df.select(
                    "SK_ID_CURR", "CODE_GENDER", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
                    "AMT_CREDIT", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
                    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "DAYS_BIRTH",
                    "DAYS_EMPLOYED", "risk_score", "prediction"
                ).first()

                # Calculer l'âge et les années d'emploi
                age = -int(prediction_row["DAYS_BIRTH"] / 365.25) if prediction_row["DAYS_BIRTH"] is not None else None
                years_employed = -float(prediction_row["DAYS_EMPLOYED"] / 365.25) if prediction_row["DAYS_EMPLOYED"] is not None and prediction_row["DAYS_EMPLOYED"] != 365243 else 0

                # 7. Former la réponse
                response = {{
                    "client_id": prediction_row["SK_ID_CURR"],
                    "client_info": {{
                        "gender": prediction_row["CODE_GENDER"],
                        "age": age,
                        "children_count": int(prediction_row["CNT_CHILDREN"]) if prediction_row["CNT_CHILDREN"] is not None else None,
                        "income": float(prediction_row["AMT_INCOME_TOTAL"]) if prediction_row["AMT_INCOME_TOTAL"] is not None else None,
                        "credit_amount": float(prediction_row["AMT_CREDIT"]) if prediction_row["AMT_CREDIT"] is not None else None,
                        "income_type": prediction_row["NAME_INCOME_TYPE"],
                        "education_type": prediction_row["NAME_EDUCATION_TYPE"],
                        "family_status": prediction_row["NAME_FAMILY_STATUS"],
                        "housing_type": prediction_row["NAME_HOUSING_TYPE"],
                        "years_employed": years_employed
                    }},
                    "prediction": {{
                        "risk_score": float(prediction_row["risk_score"]) if prediction_row["risk_score"] is not None else None,
                        "predicted_default": int(prediction_row["prediction"]) if prediction_row["prediction"] is not None else None
                    }}
                }}

                # print("[INFO] Prédiction complétée")

                return response

            except Exception as e:
                import traceback
                error_msg = f"Erreur: {{str(e)}}\\n{{traceback.format_exc()}}"
                print(error_msg)
                return {{"error": error_msg}}

        # Appeler la fonction de prédiction avec l'ID client spécifié
        result = predict_for_client({0})
        result
        """.format(client_id)

        # Exécuter la commande
        execute_url = f"{DATABRICKS_HOST}/api/1.2/commands/execute"
        execute_payload = {
            "clusterId": CLUSTER_ID,
            "contextId": context_id,
            "language": "python",
            "command": command
        }

        response = requests.post(execute_url, headers=headers, json=execute_payload)
        command_id = response.json().get("id")
        print(f"Commande exécutée avec ID: {command_id}")

        # Attendre et obtenir le résultat
        status_url = f"{DATABRICKS_HOST}/api/1.2/commands/status"
        status_params = {
            "clusterId": CLUSTER_ID,
            "contextId": context_id,
            "commandId": command_id
        }

        max_retries = 90
        retry_count = 0

        while retry_count < max_retries:
            response = requests.get(status_url, headers=headers, params=status_params)

            status_data = response.json()
            status = status_data.get("status")

            if status == "Finished":
                # Commande terminée, extraire le résultat
                result_data = status_data.get("results", {}).get("data").replace("\n", "").replace("'", '"')
                print(f"Résultat obtenu: {result_data[:100]}..." if result_data else "Pas de résultat")

                if result_data:
                    try:
                        return json.loads(result_data)
                    except Exception as e:
                        return {"error": f"Impossible de parser le résultat: {str(e)}", "raw_data": result_data[:500]}
                else:
                    return {"error": "Aucun résultat retourné"}

            elif status == "Error":
                error_message = status_data.get("results", {}).get("summary", "Erreur inconnue")
                return {"error": f"Erreur d'exécution: {error_message}"}

            # Attendre avant de vérifier à nouveau
            time.sleep(1)
            retry_count += 1

        return {"error": "Délai d'attente dépassé"}

    except Exception as e:
        import traceback
        return {"error": f"Exception: {str(e)}\n{traceback.format_exc()}"}
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
    
    
    # Appel à Databricks avec les paramètres
    # data_viz_result = run_databricks_job(job_id, notebook_params)
    data_viz_result = get_dataviz_from_databricks(analysis_type, min_credit, max_credit, min_income, CONTEXT_ID)
    
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
    print(data_viz_result)
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
    client_id = request.args.get('client_id', type=str)
    
    if not client_id:
        return jsonify({"error": "Le paramètre client_id est requis"}), 400
    
    # Paramètres pour le notebook
    notebook_params = {
        "client_id": client_id
    }
    
    try:
        # Appel à Databricks avec les paramètres
        # prediction_result = run_databricks_job(job_id, notebook_params)
        prediction_result = predict_default_from_databricks(client_id=client_id, context_id=CONTEXT_ID)        
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
    app.run(debug=True, host='0.0.0.0', port=5000)