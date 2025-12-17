import streamlit as st
import requests
import os
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd
import streamlit.components.v1 as components

load_dotenv()

st.set_page_config(page_title="Analyse de Données", page_icon="📈", layout="wide")

# ================== CSS PERSONNALISÉ ==================
st.markdown("""
<style>
    .main { background-color: #ffffff; }
    
    /* TOP METRICS ROW (Violet background) */
    .metrics-container {
        display: flex;
        justify-content: space-around;
        background-color: #E6E6FA; /* Violet clair */
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #dcdce6;
    }
    .metric-item {
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #5A5A8F;
    }
    .metric-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
    }

    /* FILTERS ACTIVE BOX */
    .active-filters {
        background-color: #E6E6FA;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: #5A5A8F;
        font-size: 14px;
    }
    
    /* CHART TITLE BAR */
    .chart-header {
        border-bottom: 2px solid #eee;
        padding-bottom: 10px;
        margin-bottom: 20px;
        font-size: 20px;
        font-weight: bold;
        color: #2C3E50;
    }

    /* KPI CARDS (White with shadow) */
    .kpi-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #eee;
        text-align: center;
    }
    .kpi-title { font-size: 14px; color: #888; margin-bottom: 5px; }
    .kpi-val { font-size: 22px; font-weight: bold; color: #2C3E50; }
    .kpi-sub { font-size: 12px; }
    .green { color: green; } .red { color: red; }

    /* RECOMMENDATION GRID (Violet Box split) */
    .reco-grid-container {
        background-color: #D6D6F5;
        border-radius: 10px;
        padding: 20px;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-top: 20px;
    }
    .reco-col h4 {
        color: #5A5A8F;
        font-size: 16px;
        margin-bottom: 15px;
    }
    .reco-col ul {
        color: #4A4A6A;
        font-size: 14px;
        padding-left: 20px;
    }
    .reco-col li { margin-bottom: 8px; }

</style>
""", unsafe_allow_html=True)

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("Analyses & Intelligence Décisionnelle")
    st.markdown("---")
    
    st.markdown("#### 📈 THÉMATIQUE D'ANALYSE")
    
    # Liste avec icônes exactes
    options = [
        "👶 👴 Âge client et défaut",
        "💰 💸 Ratio crédit/revenu",
        "💼 💵 Type de revenu",
        "🗓️ 👔 Ancienneté d'emploi", # Selection par défaut
        "📊 📄 Demandes de crédit",
        "🏠 📝 Contrat et propriété",
        "👨‍👩‍👧 👶 Famille et enfants",
        "🗓️ 🔄 Jour de demande"
    ]
    
    analysis_choice = st.radio("Choix :", options, index=3, label_visibility="collapsed")
    
    # Mapping selection -> ID pour API
    mapping = {opt: str(i+1) for i, opt in enumerate(options)}
    selected_id = mapping[analysis_choice]

    st.markdown("#### ⚙️ FILTRES D'ANALYSE")
    
    min_credit = st.slider("Montant min. crédit (€)", 0, 500000, 100000)
    max_credit = st.slider("Montant max. crédit (€)", 0, 1000000, 630000)
    min_income = st.slider("Revenu annuel min. (€)", 0, 200000, 30000)
    
    with st.expander("Filtres avancés"):
        st.write("Autres filtres...")
        
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("❓ Aide & Ressources"):
        st.write("Aide...")
        
    # Bouton Bleu foncé
    if st.button("📊 Générer l'analyse", type="primary"): 
        st.session_state['run_analysis'] = True

# ================== MAIN ==================

# 1. Bannière de mise à jour
st.info("🕒 Dernière mise à jour : 13/04/2025 à 15:41")

# 2. Métriques globales (Boite Violette)
st.markdown("""
<div class="metrics-container">
    <div class="metric-item">
        <div class="metric-value">12 583</div>
        <div class="metric-label">Clients Analysés</div>
    </div>
    <div class="metric-item">
        <div class="metric-value">5,2%</div>
        <div class="metric-label">Taux de Défaut Moyen</div>
    </div>
    <div class="metric-item">
        <div class="metric-value">0,42</div>
        <div class="metric-label">Ratio Crédit/Revenu Moyen</div>
    </div>
    <div class="metric-item">
        <div class="metric-value">385 K€</div>
        <div class="metric-label">Montant Moyen Crédit</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.info("ℹ️ Explorez les facteurs qui influencent le risque de défaut...")

# 3. Filtres Actifs (Boite Violette)
st.markdown(f"""
<div class="active-filters">
    <strong>🔻 Filtres actifs</strong><br><br>
    Min. crédit: {min_credit:,} €<br>
    Max. crédit: {max_credit:,} €<br>
    Min. revenu: {min_income:,} €
</div>
""", unsafe_allow_html=True)

# Titre dynamique
clean_title = analysis_choice.split(" ", 2)[2] # Enlever les emojis
st.markdown(f"<div class='chart-header'>↘ {clean_title} {analysis_choice.split(' ')[0]}</div>", unsafe_allow_html=True)

# 4. CHARGEMENT GRAPHIQUE (Plotly)
if st.session_state.get('run_analysis'):
    api_url = os.getenv("FLASK_API_URL")
    job_id = os.getenv("DATAVIZ_JOB_ID")
    
    with st.spinner("Chargement Databricks..."):
        try:
            # Appel API
            resp = requests.get(f"{api_url}/get_dataviz", params={
                "job_id": job_id, "analysis_type": selected_id,
                "min_credit": min_credit, "max_credit": max_credit, "min_income": min_income
            })
            
            # Affichage HTML direct si retour API OK
            if resp.status_code == 200:
                # Flask renvoie le HTML complet du graph
                components.html(resp.text, height=500)
            else:
                st.error("Erreur API Viz")
        except Exception as e:
            st.error(str(e))
else:
    st.write("Cliquez sur 'Générer l'analyse' pour voir le graphique.")

# 5. INDICATEURS CLÉS (3 Cartes Blanches)
st.markdown("### 📊 Indicateurs clés")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">Ancienneté à risque max</div>
        <div class="kpi-val">&lt; 1 an</div>
        <div class="kpi-sub red">↑ 7.9%</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">Ancienneté à risque min</div>
        <div class="kpi-val">&gt; 10 ans</div>
        <div class="kpi-sub green">↓ 3.2%</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">Ancienneté moyenne</div>
        <div class="kpi-val">5.8 ans</div>
        <div class="kpi-sub green">↑ 0.3 ans</div>
    </div>
    """, unsafe_allow_html=True)

# 6. INTERPRÉTATION & RECOMMANDATIONS (Boite Violette divisée)
st.markdown("### ⓘ Interprétation")
st.write("Une stabilité professionnelle plus longue est associée à un risque plus faible...")

st.markdown("### 📄 Recommandations")
st.markdown("""
<div class="reco-grid-container">
    <div class="reco-col">
        <h4>Recommandations opérationnelles</h4>
        <ul>
            <li>Adapter les exigences de garantie en fonction du profil.</li>
            <li>Proposer des taux d'intérêt ajustés au risque.</li>
            <li>Mettre en place un suivi spécifique pour les segments risqués.</li>
            <li>Réviser régulièrement les critères d'octroi.</li>
        </ul>
    </div>
    <div class="reco-col" style="border-left: 1px solid rgba(0,0,0,0.1); padding-left: 20px;">
        <h4>Actions stratégiques</h4>
        <ul>
            <li>Intégrer ces variables dans les modèles de scoring.</li>
            <li>Développer des parcours client différenciés.</li>
            <li>Établir un reporting mensuel sur l'évolution.</li>
            <li>Organiser des sessions de formation.</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)