import streamlit as st
import pandas as pd
import joblib

model = joblib.load('credit_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="CréditScore — Aide à la décision", page_icon="💳")
st.title("💳 CréditScore")
st.caption("Outil d'aide à la décision pour l'octroi de crédit — modèle prédictif basé sur l'historique de solvabilité")

st.divider()

# Taux de conversion approximatif DM -> XOF pour adapter le dataset historique au contexte local
DM_TO_XOF = 350

col1, col2 = st.columns(2)

with col1:
    checking = st.selectbox(
        "Statut du compte courant",
        ["Aucun compte", "Solde négatif", "Solde entre 0 et 130 000 F", "Solde supérieur à 130 000 F"],
        help="Situation actuelle du compte bancaire principal du client au moment de la demande."
    )
    duration = st.slider(
        "Durée du crédit (mois)", 4, 72, 24,
        help="Durée totale sur laquelle le crédit sera remboursé, exprimée en mois."
    )
    credit_history = st.selectbox(
        "Historique de crédit",
        ["Aucun crédit / tous remboursés", "Crédits remboursés à temps",
         "Retards passés", "Crédit en cours ailleurs", "Crédit critique / retards"],
        help="Comportement de remboursement observé sur les crédits précédents du client."
    )
    purpose = st.selectbox(
        "Objet du crédit",
        ["Voiture neuve", "Voiture occasion", "Meubles", "Équipement audio/vidéo",
         "Électroménager", "Réparations", "Éducation", "Business", "Autre"],
        help="Usage prévu des fonds empruntés."
    )
    credit_amount = st.number_input(
        "Montant du crédit (F CFA)", 
        min_value=int(250 * DM_TO_XOF), 
        max_value=int(20000 * DM_TO_XOF), 
        value=int(3000 * DM_TO_XOF), 
        step=5000,
        help="Montant total demandé par le client."
    )
    savings = st.selectbox(
        "Épargne disponible",
        ["Aucune épargne", "Moins de 35 000 F", "Entre 35 000 et 175 000 F",
         "Entre 175 000 et 350 000 F", "Plus de 350 000 F"],
        help="Montant approximatif d'épargne détenue par le client, hors compte courant."
    )

with col2:
    employment = st.selectbox(
        "Ancienneté dans l'emploi actuel",
        ["Sans emploi", "Moins d'1 an", "Entre 1 et 4 ans", "Entre 4 et 7 ans", "7 ans ou plus"],
        help="Stabilité professionnelle : durée depuis laquelle le client occupe son poste actuel."
    )
    age = st.slider(
        "Âge du client", 18, 75, 35,
        help="Âge du demandeur au moment de la demande de crédit."
    )
    housing = st.selectbox(
        "Situation de logement",
        ["Locataire", "Propriétaire", "Logé gratuitement"],
        help="Statut résidentiel actuel du client."
    )
    job = st.selectbox(
        "Catégorie professionnelle",
        ["Sans emploi / non qualifié", "Employé qualifié", "Cadre / indépendant qualifié"],
        help="Niveau de qualification du poste occupé, indicateur de stabilité de revenu."
    )
    installment_rate = st.slider(
        "Taux de mensualité (% du revenu)", 1, 4, 2,
        help="Part approximative du revenu mensuel consacrée au remboursement de ce crédit."
    )
    existing_credits = st.slider(
        "Nombre de crédits déjà en cours", 1, 4, 1,
        help="Nombre total de crédits actifs détenus par le client, celui-ci inclus."
    )

st.divider()

map_checking = {"Aucun compte": "A14", "Solde négatif": "A11", 
                 "Solde entre 0 et 130 000 F": "A12", "Solde supérieur à 130 000 F": "A13"}
map_history = {"Aucun crédit / tous remboursés": "A30", "Crédits remboursés à temps": "A31",
               "Retards passés": "A32", "Crédit en cours ailleurs": "A33", "Crédit critique / retards": "A34"}
map_purpose = {"Voiture neuve": "A40", "Voiture occasion": "A41", "Meubles": "A42", "Équipement audio/vidéo": "A43",
               "Électroménager": "A44", "Réparations": "A45", "Éducation": "A46", "Business": "A49", "Autre": "A410"}
map_savings = {"Aucune épargne": "A65", "Moins de 35 000 F": "A61", "Entre 35 000 et 175 000 F": "A62",
               "Entre 175 000 et 350 000 F": "A63", "Plus de 350 000 F": "A64"}
map_employment = {"Sans emploi": "A71", "Moins d'1 an": "A72", "Entre 1 et 4 ans": "A73", 
                   "Entre 4 et 7 ans": "A74", "7 ans ou plus": "A75"}
map_housing = {"Locataire": "A151", "Propriétaire": "A152", "Logé gratuitement": "A153"}
map_job = {"Sans emploi / non qualifié": "A171", "Employé qualifié": "A173", "Cadre / indépendant qualifié": "A174"}

if st.button("🔍 Évaluer la demande", use_container_width=True):
    input_data = {
        'checking_account_status': map_checking[checking],
        'duration_months': duration,
        'credit_history': map_history[credit_history],
        'purpose': map_purpose[purpose],
        'credit_amount': credit_amount / DM_TO_XOF,  # reconversion vers l'échelle d'entraînement du modèle
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

    st.caption("⚠️ Outil de démonstration à but pédagogique, basé sur des données historiques (German Credit / UCI) adaptées au contexte local. Ne constitue pas une décision de crédit réelle.")