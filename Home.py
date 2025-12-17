import streamlit as st

st.set_page_config(page_title="Risk Banking", page_icon="🏦", layout="wide")

st.title("🏦 Risk Banking App")

st.markdown("""
### Bienvenue

Cette application utilise une architecture hybride :
1. **MongoDB** : Pour récupérer les données d'identité (Nom, Photo).
2. **Databricks** : Pour calculer le score de risque en temps réel.

👈 **Utilisez le menu à gauche pour naviguer.**

* **👤 Prediction Client** : Entrez un ID (ex: `114843`) pour voir le dossier complet.
* **📈 Data Analysis** : Pour voir les statistiques globales.
""")