import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from PIL import Image

from externalFunctions import getDataframe, retirerColonnes, renameDataframe, rolling_averages, plot_confusion_matrix, plot_feature_importances


def prepare_future_match(home_team, away_team, date, time, rolling_stats, odds):
    # Créer un DataFrame pour le match futur
    future_match = pd.DataFrame({
        'Equipe_Domicile': [home_team],
        'Equipe_Exterieure': [away_team],
        'Date_Match': [pd.to_datetime(date)],
        'Time': [time],
        'opp_code': [matches['Equipe_Exterieure'].astype('category').cat.codes[matches['Equipe_Exterieure'] == away_team].values[0]],
        'Jour_Match': [pd.to_datetime(date).dayofweek],
        'Heure_Match': [int(time.split(':')[0])],  # Extraire l'heure
        'B365H': [odds[0]],  # Cote domicile
        'B365D': [odds[1]],  # Cote match nul
        'B365A': [odds[2]]   # Cote extérieure
    })

    home_stats = rolling_stats[rolling_stats['Equipe_Domicile'] == home_team].iloc[-1]
    away_stats = rolling_stats[rolling_stats['Equipe_Domicile'] == away_team].iloc[-1]

    # Ajouter les caractéristiques des équipes
    for col in new_cols:
        future_match[col] = [home_stats[col]]  # Utiliser les stats de l'équipe à domicile
    for col in new_features:
        future_match[col] = [home_stats[col]]  # Utiliser les stats de l'équipe à domicile

    return future_match

def predict_match(home_team, away_team, date, time, odds):
    # Préparer les données du match futur
    future_match = prepare_future_match(home_team, away_team, date, time, matches_rolling, odds)
    prediction = rf.predict(future_match[predictors])

    result_map = {1: "Victoire Domicile", 0: "Victoire Extérieure", 2: "Match Nul"}
    predicted_result = result_map[prediction[0]]

    return predicted_result

matches = getDataframe()
matches = retirerColonnes(matches)
matches = renameDataframe(matches)

# Convertir les équipes en codes numériques
matches['opp_code'] = matches['Equipe_Exterieure'].astype('category').cat.codes

matches['Jour_Match'] = matches['Date_Match'].dt.dayofweek

# Gérer les heures du match
matches['Heure_Match'] = matches['Time'].str.replace(":.+", "", regex=True)
matches['Heure_Match'] = matches['Heure_Match'].astype(float)
matches['Heure_Match'].fillna(matches['Heure_Match'].mean(), inplace=True)
matches['Heure_Match'] = matches['Heure_Match'].astype(int)

# Variable cible : 1 si l'équipe à domicile gagne, 0 si l'équipe à l'extérieur gagne, 2 pour un match nul
matches["Target_Prediction"] = matches["Resultat"].map({"H": 1, "A": 0, "D": 2})

cols = ["Buts_Domicile", "Buts_Exterieur", "Tirs_Domicile", "Tirs_Exterieur", "Tirs_Cadres_Domicile", "Tirs_Cadres_Exterieur", "Corners_Domicile", "Corners_Exterieur"]
new_cols = [f"{c}_rolling" for c in cols]


matches_rolling = matches.groupby("Equipe_Domicile").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('Equipe_Domicile')
matches_rolling.index = range(matches_rolling.shape[0])

# Sélection des variables pour le modèle
new_features = [
    "Buts_Domicile", "Buts_Exterieur",
    "Tirs_Domicile", "Tirs_Exterieur",
    "Tirs_Cadres_Domicile", "Tirs_Cadres_Exterieur",
    "Corners_Domicile", "Corners_Exterieur",
    "Fautes_Domicile", "Fautes_Exterieur",
    "Cartons_Jaunes_Domicile", "Cartons_Jaunes_Exterieur",
    "B365H", "B365D", "B365A"  # Cotes des bookmakers
]

predictors = ["opp_code", "Jour_Match", "Heure_Match", 'B365H', 'B365D', 'B365A'] + new_features + new_cols

# Division des données en train/test
train = matches_rolling[matches_rolling["Saison"] <= "2020"]
test = matches_rolling[matches_rolling["Saison"] >= "2021"]

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=14,
    min_samples_split=14,
    min_samples_leaf=2,
    random_state=1,
    class_weight='balanced'
    # n_estimators=50,
    # min_samples_split=10,
    # random_state=1,
)

rf.fit(train[predictors], train["Target_Prediction"])
preds = rf.predict(test[predictors])

# Évaluation du modèle
accuracy = accuracy_score(test["Target_Prediction"], preds)
conf_matrix = confusion_matrix(test["Target_Prediction"], preds)
precision = precision_score(test["Target_Prediction"], preds, average='weighted')  # Moyenne pondérée pour chaque classe


# Affichage des résultats
plot_confusion_matrix(conf_matrix)
# Calculer les importances des caractéristiques
importances = rf.feature_importances_
feature_names = predictors
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})


# Générer et enregistrer le graphique des importances des caractéristiques
plot_feature_importances(feature_importance_df)

print(f"Taille de l'echantillon de train : {train.shape[0]}")
print(f"Taille de l'echantillon de test : {test.shape[0]}")
print(f"Pourcentage de test : {test.shape[0] / (train.shape[0] + test.shape[0]) * 100:.2f}%")

print(f"✅ Précision du modèle : {accuracy:.4f}") # proportion de prédictions correctes parmi toutes les prédictions effectuées
print(f"🎯 Score de précision : {precision:.4f}") # se concentre spécifiquement sur la qualité des prédictions positives. Il est défini comme le rapport entre le nombre de vrais positifs (TP) et le nombre total de prédictions positives (TP + FP)
print(f"📊 Matrice de confusion :\n {conf_matrix}")



st.set_page_config(
    page_title="Machine Learning - Ligue 1",
    page_icon="⚽",
)


st.markdown("<h1 style='font-size: 36px; font-weight: bolder;'>Machine Learning - Matchs de Ligue 1</h1>", unsafe_allow_html=True)

# Informations sur le modèle
st.markdown("<h1 style='font-size: 24px; font-weight: bolder;'>Informations sur le modèle actuel (RandomForestClassifier)</h1>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.metric("✅ Précision (modèle)", f"{accuracy:.2f}")
    st.metric("Nb valeurs pour l'entrainement", f"{train.shape[0]}")
with col2:
    st.metric("🎯 Score de précision (modèle)", f"{precision:.2f}")  # Assurez-vous d'utiliser la variable 'precision' ici
    st.metric("Nb valeurs pour les tests", f"{test.shape[0]} ({test.shape[0] / (train.shape[0] + test.shape[0]) * 100:.2f}%)")

st.markdown("<h1 style='font-size: 24px; font-weight: bolder;'>📊 Matrice de confusion</h1>", unsafe_allow_html=True)
st.image('matrice_confusion.png', use_container_width=True)
st.markdown("<h1 style='font-size: 24px; font-weight: bolder;'>Importances des caractéristiques</h1>", unsafe_allow_html=True)
st.image('feature_importances.png', use_container_width=True)

equipes = matches['Equipe_Domicile'].unique()

st.markdown("<h2>⚽ Choisissez les équipes qui s'affrontent</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    equipe_domicile = st.selectbox("Équipe Domicile", equipes)
with col2:
    equipe_exterieure = st.selectbox("Équipe Extérieure", equipes)


st.markdown("<h2>📊 Entrez les cotes</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    cote_domicile = st.text_input("Cote Domicile (ex: 1.5)", "")
with col2:
    cote_nul = st.text_input("Cote Match Nul (ex: 1.5)", "")
with col3:
    cote_exterieure = st.text_input("Cote Extérieure (ex: 1.5)", "")

st.markdown("<h2>📅 Entrez la date du match</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    date_match = st.date_input("Date du Match", pd.to_datetime("today"))
with col2:
    time_match = st.time_input("Heure du Match", value=pd.to_datetime("12:00").time())



if st.button("Prédire le résultat du match"):
    odds = [
        float(cote_domicile) if cote_domicile else 1.0,
        float(cote_nul) if cote_nul else 1.0,
        float(cote_exterieure) if cote_exterieure else 1.0
    ]

    predicted_result = predict_match(equipe_domicile, equipe_exterieure, date_match, time_match.strftime("%H:%M"), odds)

    st.write(f"Le résultat prédit pour le match {equipe_domicile} (Domicile) contre {equipe_exterieure} est : {predicted_result}")
