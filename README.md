🏦 Risk Banking - Plateforme d'Analyse de Risque de Crédit

Application bancaire d'aide à la décision combinant Machine Learning (Databricks), Base de données NoSQL (MongoDB) et Interface Web interactive (Streamlit).

📋 Description du Projet

Risk Banking est une application web interactive conçue pour les institutions financières. Elle vise à moderniser et faciliter le processus de décision d'octroi de crédit grâce à l'intelligence artificielle et à la visualisation de données.

### Objectifs
- Réduire les risques financiers en anticipant les défauts de paiement.
- Fournir des outils visuels pour le suivi des clients et des portefeuilles.
- Faciliter la prise de décision basée sur les données.

🏗️ Architecture Technique


| Composant          | Technologie / Outils                              |
|------------------- |---------------------------------------------------|
| Frontend           | [Streamlit](https://streamlit.io/)                |
| Backend / API      | [Flask](https://flask.palletsprojects.com/)       |
| Calcul & ML        | [Databricks](https://databricks.com/) (PySpark)   |
| Base de Données    | [MongoDB Atlas](https://www.mongodb.com/atlas)    |


3. Configurer les variables d'environnement

Créer un fichier .env à la racine :

# Flask API
FLASK_API_URL=http://localhost:5000
# Databricks
DATABRICKS_INSTANCE=https://adb-xxxx.xx.azuredatabricks.net
DATABRICKS_TOKEN=dapi...
CLUSTER_ID = "1216-092113-xxxxx"
PREDICT_JOB_ID=1001516646288990
DATAVIZ_JOB_ID=847349130442312
ONNECTION_STRING=InstrumentationKey=e291b322-28a6-4.....
# MongoDB
MONGODB_URI=mongodb+srv://sdv_user:SDV2025@cluster0.t2ptc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
MONGO_DB_NAME="default_risk" 
MONGO_COLLECTION_NAME="users_data"

4. Lancer l'application

Backend Flask : python app.py
Frontend Streamlit : streamlit run Home.py

🖥️ Guide d'Utilisation

Accueil : Page principale avec présentation générale et navigation.
Prédiction Client :
    -Sélectionner un client.
   -Obtenir la prédiction du risque de défaut.
   -Visualiser les informations détaillées du client.
Analyse des Données :
   -Graphiques interactifs des tendances du portefeuille.
   -Filtrage par segment, produit ou niveau de risque.

📂 Organisation du Projet
Cloud_Bigdata/
├── .env                     # Variables de configuration
├── app.py                   # Backend Flask
├── Home.py                  # Accueil Streamlit
├── requirements.txt         # Dépendances Python
└── pages/
    ├── 1_👤_Prediction_Client.py
    └── 2_📈_Data_Analysis.py

👥 Auteurs

Développé pour le projet Big Data & Cloud par :
Ahmed PEKASSA
Bramly MBAKOP

📖 Références

Streamlit Documentation (https://docs.streamlit.io/)

Flask Documentation (https://flask.palletsprojects.com/)

Databricks Guide (https://docs.databricks.com/)

MongoDB Atlas Documentation (https://www.mongodb.com/docs/atlas/)
