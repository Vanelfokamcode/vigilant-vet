import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Vigilant-Vet Orchestrator", layout="wide")

# Custom CSS for a professional dark look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Vigilant-Vet Orchestrator")
st.markdown("### Système Multi-Agents RAG pour la Conformité Vétérinaire")

# Sidebar for settings
with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API URL", value="http://localhost:8000/query")
    st.info("Ce système interroge les RCP officiels de l'ANMV et audite chaque réponse via un agent Garde-Fou.")

query = st.text_input("Posez votre question réglementaire :", placeholder="ex: quelle dose amoxicilline pour un porc de 80kg ?")

if query:
    with st.spinner("Les agents analysent les notices ANMV..."):
        try:
            response = requests.post(api_url, json={"question": query})
            response.raise_for_status()
            data = response.json()

            # Main Results
            col1, col2, col3 = st.columns(3)
            col1.metric("Score de Confiance", f"{data['audit_score']*100:.0f}%")
            col2.metric("Itérations Agent", data["iterations"])
            col3.metric("Calcul", "Validé ✅" if data["calculation"] else "N/A")

            st.markdown("---")
            st.subheader("Réponse Validée")
            st.success(data["answer"])

            # Tabs for details
            tab1, tab2, tab3 = st.tabs(["📊 Détails Techniques", "📚 Sources ANMV", "🔍 Traces Agents"])
            
            with tab1:
                st.write("**Calcul effectué par l'Agent 02 :**")
                st.code(data["calculation"] if data["calculation"] else "Aucun calcul requis ou données manquantes")
            
            with tab2:
                for src in data["sources"]:
                    with st.expander(f"Source : {src['product_name']} (Espèce : {src['species']})"):
                        st.write(f"**Pertinence (Score RRF) :** {src['score']}")
                        st.write(f"**Extrait :** ...{src['excerpt']}...")

            with tab3:
                st.json(data)

        except Exception as e:
            st.error(f"Erreur de connexion à l'API : {e}")

st.markdown("---")
st.caption("Vanel FOKAM — Vigilant Infrastructure — 2026")
