import streamlit as st
import requests
import os
import pymongo
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

# Charger les variables
load_dotenv()

st.set_page_config(page_title="Prédiction Client", page_icon="🔮", layout="wide")

# ================== CSS PERSONNALISÉ (Pour matcher le design) ==================
st.markdown("""
<style>
    /* Style global */
    .main { background-color: #ffffff; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; color: #2C3E50; }
    
    /* PROFIL CLIENT CENTRÉ */
    .profile-container {
        text-align: center;
        padding: 20px;
        background-color: #E6E6FA; /* Violet très clair */
        border-radius: 10px;
        width: fit-content;
        margin: 0 auto;
    }
    .profile-img {
        width: 150px;
        height: 150px;
        border-radius: 50%; /* Rond parfait */
        object-fit: cover;
        border: 4px solid #E6E6FA;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .client-name {
        color: #3b5998;
        font-size: 24px;
        font-weight: bold;
        margin-top: 10px;
    }
    .client-id {
        color: #888;
        font-size: 14px;
    }

    /* ALERT BOX (Simulation de risque) */
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #20a08d;
        padding: 15px;
        border-radius: 5px;
        color: #2c3e50;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* RECOMMANDATION BOX (Violette) */
    .reco-box {
        background-color: #D6D6F5; /* Violet capture d'écran */
        padding: 25px;
        border-radius: 8px;
        color: #4A4A6A;
        margin-top: 10px;
    }
    .reco-title {
        font-size: 22px;
        font-weight: bold;
        color: #5A5A8F;
        margin-bottom: 10px;
    }
    .reco-text {
        font-size: 16px;
        margin-bottom: 20px;
    }
    .reco-stats {
        display: flex;
        justify-content: space-between;
        font-size: 14px;
        color: #666;
        border-top: 1px solid rgba(0,0,0,0.1);
        padding-top: 10px;
    }

    /* BADGE DE RISQUE */
    .risk-badge {
        background-color: #a8e6cf; /* Vert pastel */
        color: #1b5e20;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        width: fit-content;
        margin: 20px auto 0 auto;
        border: 2px solid #fff;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .risk-badge-high {
        background-color: #ff8b94; /* Rouge pastel */
        color: #800000;
    }

    /* JAUGE LINEAIRE CSS */
    .gauge-container {
        width: 100%;
        margin: 30px 0;
    }
    .gauge-bar {
        height: 10px;
        width: 100%;
        background: linear-gradient(90deg, #4CAF50 0%, #FFC107 50%, #F44336 100%);
        border-radius: 5px;
        position: relative;
    }
    .gauge-cursor {
        width: 20px;
        height: 20px;
        background-color: white;
        border: 3px solid #3b5998;
        border-radius: 50%;
        position: absolute;
        top: -5px;
        transform: translateX(-50%);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .gauge-labels {
        display: flex;
        justify-content: space-between;
        color: #888;
        font-size: 12px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- FONCTIONS ---
def get_client_mongo(client_id):
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGO_DB_NAME", "credit_db")
    col_name = os.getenv("MONGO_COLLECTION_NAME", "clients")
    
    if not uri: return None
    try:
        client = pymongo.MongoClient(uri)
        db = client[db_name]
        col = db[col_name]
        try: q_id = int(client_id)
        except: q_id = client_id
        return col.find_one({"SK_CURR_ID": q_id})
    except Exception as e:
        return None
    finally:
        if 'client' in locals(): client.close()

def get_prediction_api(client_id):
    api_url = os.getenv("FLASK_API_URL")
    job_id = os.getenv("PREDICT_JOB_ID")
    if not api_url: return None
    try:
        response = requests.get(f"{api_url}/predict_default", params={"client_id": client_id, "job_id": job_id})
        return response.json()
    except: return None

# ================== SIDEBAR ==================
with st.sidebar:
    st.image("https://github.com/archiducarmel/SupdeVinci_BigData_Cloud/releases/download/datas/risk_banking.jpg", width=180) # Logo RB
    
    st.markdown("### 🔍 Rechercher un client")
    
    input_id = st.text_input("ID Client", "118893")
    
    # Bouton Rouge (via type primary et thème ou CSS)
    btn_search = st.button("🔍 Analyser le risque", type="primary")
    
    with st.expander("❓ Aide & Ressources"):
        st.write("Documentation disponible ici.")

# ================== MAIN CONTENT ==================
st.title("🔮 Prédiction de Défaut Client")

st.markdown("""
<div class="info-box">
    📊 <strong>SIMULATION DE RISQUE</strong> - Cette page permet d'estimer la probabilité de défaut d'un client en fonction de son ID client. Entrez un ID client pour obtenir une analyse complète du risque.
</div>
""", unsafe_allow_html=True)

if btn_search and input_id:
    # 1. Mongo
    identity = get_client_mongo(input_id)
    
    # 2. Centrage du Profil (Image + Nom) comme sur l'image 1
    col_spacer_l, col_profile, col_spacer_r = st.columns([1, 2, 1])
    
    with col_profile:
        if identity:
            photo = identity.get("PhotoURL", "https://www.w3schools.com/howto/img_avatar.png")
            fname = identity.get("FirstName", "Inconnu")
            lname = identity.get("LastName", "")
            
            # HTML brut pour centrer parfaitement l'image ronde et le texte
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 30px;">
                <img src="{photo}" class="profile-img">
                <div class="client-name">{fname} {lname}</div>
                <div class="client-id">ID Client: {input_id}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Client introuvable dans MongoDB")

    # 3. Prédiction API
    with st.spinner("Analyse en cours..."):
        api_res = get_prediction_api(input_id)

    if api_res and "error" not in api_res:
        pred = api_res.get("prediction", {})
        info = api_res.get("client_info", {})
        risk_score = pred.get("risk_score", 0.0) # ex: 0.317
        
        # --- BLOC RÉSULTAT (Image 2) ---
        st.markdown(f"### 📊 Résultat de l'analyse de risque")
        
        # Jauge Linéaire Custom (HTML/CSS)
        risk_percent = risk_score * 100
        cursor_left = f"{risk_percent}%"
        
        st.markdown(f"""
        <div class="gauge-container">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>Faible</span>
                <span style="font-weight: bold; font-size: 18px; color: #FFA500;">{risk_percent:.1f}%</span>
            </div>
            <div class="gauge-bar">
                <div class="gauge-cursor" style="left: {cursor_left};"></div>
            </div>
            <div class="gauge-labels">
                <span>0%</span>
                <span>100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Boite de Recommandation Violette
        reco_text = "Acceptation recommandée. Le profil présente un faible risque de défaut." if risk_score < 0.5 else "Risque élevé détecté. Examen manuel requis."
        date_analysis = datetime.now().strftime("%d/%m/%Y")
        montant = f"{info.get('credit_amount', 0):,} €"
        ratio = f"{info.get('credit_amount', 0) / (info.get('income', 1) or 1):.2f}"
        
        st.markdown(f"""
        <div class="reco-box">
            <div class="reco-title">Recommandation</div>
            <div class="reco-text">{reco_text}</div>
            <div class="reco-stats">
                <div><strong>Demande</strong><br>{montant}</div>
                <div><strong>Ratio crédit/revenu</strong><br>{ratio}</div>
                <div><strong>Date d'analyse</strong><br>{date_analysis}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Badge en bas
        badge_class = "risk-badge" if risk_score < 0.5 else "risk-badge risk-badge-high"
        badge_text = "Risque Faible" if risk_score < 0.5 else "Risque Élevé"
        
        st.markdown(f"""
        <div class="{badge_class}">
            {badge_text}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("Erreur de prédiction")