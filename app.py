import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---- Configuration de l'interface ----
st.set_page_config(page_title="Prédiction HBP IA", layout="centered")
st.title("🔍 Prédiction de l'HBP et recommandation thérapeutique")
st.markdown("""
Entrez les données cliniques du patient ci-dessous :
""")

# ---- Formulaire utilisateur ----
with st.form("formulaire_hbp"):
    age = st.number_input("Âge (en années)", min_value=40, max_value=100, value=65)
    psa = st.number_input("PSA (ng/mL)", min_value=0.0, value=4.5)
    ipss = st.slider("Score IPSS", 0, 35, 15)
    volume = st.number_input("Volume prostatique (mL)", min_value=10.0, value=45.0)
    residu = st.number_input("Résidu post-mictionnel (mL)", min_value=0.0, value=60.0)
    submit = st.form_submit_button("Analyser")

# ---- Modèle IA simulé ----
def charger_modele():
    # Données simulées pour l'exemple
    np.random.seed(0)
    X_simule = pd.DataFrame({
        "Age": np.random.randint(50, 80, 300),
        "PSA": np.random.gamma(2.5, 2.0, 300),
        "IPSS": np.random.randint(0, 36, 300),
        "Volume": np.random.normal(50, 12, 300),
        "Residu": np.random.normal(60, 20, 300),
    })
    y_simule = ((X_simule["Volume"] > 40) & (X_simule["IPSS"] > 8)).astype(int)

    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_simule, y_simule)
    return model

# ---- Logique thérapeutique ----
def proposer_traitement(ipss, volume, residu):
    if ipss < 8 and volume < 40:
        return "Surveillance / hygiène de vie"
    elif 8 <= ipss <= 19 or volume >= 40:
        return "Traitement médical (Alpha-bloquant +/- 5ARI)"
    elif ipss >= 20 or residu > 100:
        return "Évaluation pour traitement chirurgical"
    else:
        return "Surveillance active"

# ---- Analyse et résultats ----
if submit:
    model = charger_modele()
    donnees = pd.DataFrame({
        "Age": [age],
        "PSA": [psa],
        "IPSS": [ipss],
        "Volume": [volume],
        "Residu": [residu],
    })

    prediction = model.predict(donnees)[0]
    traitement = proposer_traitement(ipss, volume, residu)

    st.subheader("🧠 Résultat IA")
    st.write("**HBP significative prédite :**", "Oui" if prediction == 1 else "Non")
    st.write("**Approche thérapeutique proposée :**", traitement)

    st.success("Analyse terminée.")
