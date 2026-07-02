import streamlit as st
import pandas as pd
import joblib

# Chargement des artefacts du modèle
model = joblib.load('credit_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Credit Scoring App", page_icon="💳")
st.title("💳 Évaluation de Solvabilité Crédit")
st.caption("Modèle de scoring basé sur German Credit Data (UCI) — LogisticRegression, recall optimisé sur défauts")

st.divider()

col1, col2 = st.columns(2)

with col1:
    checking = st.selectbox("Statut compte courant", 
        ["Aucun compte", "< 0 DM", "0-200 DM", ">= 200 DM"])
    duration = st.slider("Durée du crédit (mois)", 4, 72, 24)
    credit_history = st.selectbox("Historique de crédit",
        ["Aucun crédit / tous remboursés", "Crédits remboursés à temps",
         "Retards passés", "Crédit en cours ailleurs", "Crédit critique / retards"])
    purpose = st.selectbox("Objet du crédit",
        ["Voiture neuve", "Voiture occasion", "Meubles", "Radio/TV",
         "Électroménager", "Réparations", "Éducation", "Business", "Autre"])
    credit_amount = st.number_input("Montant du crédit (DM)", 250, 20000, 3000)
    savings = st.selectbox("Épargne",
        ["Aucune épargne", "< 100 DM", "100-500 DM", "500-1000 DM", ">= 1000 DM"])

with col2:
    employment = st.selectbox("Ancienneté emploi",
        ["Sans emploi", "< 1 an", "1-4 ans", "4-7 ans", ">= 7 ans"])
    age = st.slider("Âge", 18, 75, 35)
    housing = st.selectbox("Logement", ["Location", "Propriétaire", "Gratuit"])
    job = st.selectbox("Type d'emploi",
        ["Sans emploi / non qualifié", "Employé qualifié", "Cadre / indépendant qualifié"])
    installment_rate = st.slider("Taux de mensualité (% du revenu)", 1, 4, 2)
    existing_credits = st.slider("Nombre de crédits en cours", 1, 4, 1)

st.divider()

# Mapping vers les codes originaux du dataset (nécessaires pour matcher l'encodage du modèle)
map_checking = {"Aucun compte": "A14", "< 0 DM": "A11", "0-200 DM": "A12", ">= 200 DM": "A13"}
map_history = {"Aucun crédit / tous remboursés": "A30", "Crédits remboursés à temps": "A31",
               "Retards passés": "A32", "Crédit en cours ailleurs": "A33", "Crédit critique / retards": "A34"}
map_purpose = {"Voiture neuve": "A40", "Voiture occasion": "A41", "Meubles": "A42", "Radio/TV": "A43",
               "Électroménager": "A44", "Réparations": "A45", "Éducation": "A46", "Business": "A49", "Autre": "A410"}
map_savings = {"Aucune épargne": "A65", "< 100 DM": "A61", "100-500 DM": "A62",
               "500-1000 DM": "A63", ">= 1000 DM": "A64"}
map_employment = {"Sans emploi": "A71", "< 1 an": "A72", "1-4 ans": "A73", "4-7 ans": "A74", ">= 7 ans": "A75"}
map_housing = {"Location": "A151", "Propriétaire": "A152", "Gratuit": "A153"}
map_job = {"Sans emploi / non qualifié": "A171", "Employé qualifié": "A173", "Cadre / indépendant qualifié": "A174"}

if st.button("🔍 Évaluer la demande", use_container_width=True):
    # Construction de la ligne avec valeurs par défaut raisonnables pour les colonnes non exposées au formulaire
    input_data = {
        'checking_account_status': map_checking[checking],
        'duration_months': duration,
        'credit_history': map_history[credit_history],
        'purpose': map_purpose[purpose],
        'credit_amount': credit_amount,
        'savings_account': map_savings[savings],
        'employment_since': map_employment[employment],
        'installment_rate': installment_rate,
        'personal_status_sex': 'A93',
        'other_debtors': 'A101',
        'residence_since': 2,
        'property': 'A121',
        'age': age,
        'other_installment_plans': 'A143',
        'housing': map_housing[housing],
        'existing_credits': existing_credits,
        'job': map_job[job],
        'num_dependents': 1,
        'telephone': 'A192',
        'foreign_worker': 'A201',
    }

    input_df = pd.DataFrame([input_data])
    cat_cols = input_df.select_dtypes(include='object').columns.tolist()
    input_encoded = pd.get_dummies(input_df, columns=cat_cols)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    input_scaled = scaler.transform(input_encoded)
    proba_default = model.predict_proba(input_scaled)[0][1]
    proba_ok = 1 - proba_default

    st.divider()
    if proba_default < 0.4:
        st.success(f"✅ Profil favorable — {proba_ok:.0%} de probabilité de remboursement")
        st.write("Le profil présente des indicateurs de stabilité financière cohérents avec un faible risque de défaut.")
    elif proba_default < 0.6:
        st.warning(f"⚠️ Profil à surveiller — {proba_ok:.0%} de probabilité de remboursement")
        st.write("Le profil présente un risque modéré. Une analyse manuelle complémentaire est recommandée.")
    else:
        st.error(f"❌ Profil à risque élevé — {proba_ok:.0%} de probabilité de remboursement")
        st.write("Le profil présente des indicateurs de risque significatifs de défaut de paiement.")

    st.caption("⚠️ Outil de démonstration à but pédagogique — ne constitue pas une décision de crédit réelle.")