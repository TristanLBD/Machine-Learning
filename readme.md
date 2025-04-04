# Prédiction des matchs de Ligue 1 - Machine Learning

Application Streamlit utilisant le machine learning afin de prédire les résultats des matchs de football de la Ligue 1.

Données utilisées retrouvable sur [football-data.co.uk](https://www.football-data.co.uk/downloadm.php) ainsi que leurs [définitions](https://www.football-data.co.uk/notes.txt).

## 🔗 Lien de l'application Streamlit :

Application consultable [ici](https://machine-learning-lbdtristan.streamlit.app/).

### Caractéristiques utilisées

-   Jour_Match
-   Heure_Match
-   B365H (Cotes bookmakers)
-   B365D (Cotes bookmakers)
-   B365A (Cotes bookmakers)
-   Buts_Domicile
-   Buts_Exterieur
-   Tirs_Domicile
-   Tirs_Exterieur
-   Tirs_Cadres_Domicile
-   Tirs_Cadres_Exterieur
-   Corners_Domicile
-   Corners_Exterieur
-   Fautes_Domicile
-   Fautes_Exterieur
-   Cartons_Jaunes_Domicile
-   Cartons_Jaunes_Exterieur

## Algorithme(s) disponible(s)

### Random Forest

Random Forest est un algorithme d’apprentissage automatique supervisé (supervised learning), utilisé pour des tâches de classification ou de régression.
C’est un ensemble (ou "ensemble learning") de plusieurs arbres de décision.

**Avantages :**

-   Très bonne précision sans trop d’effort.
-   Moins de risque de surapprentissage (overfitting) que les arbres simples.
-   Bonne gestion des données manquantes.
-   Peut gérer les variables catégorielles et numériques.

**Inconvénients :**

-   Peut être plus lent que d'autres algorithmes
-   Moins interprétable qu'un seul arbre de décision

## Structure du projet

-   `main.py` - Fichier principal de l'application
-   `externalFunctions.py` - Fonctions externalisées pour une meilleure lisibilité du code principal
-   `datas/F1_XXXX_XXXX.csv` - Fichiers CSVs contenants les données de ligue 1 servant a entrainer / tester le modèle de Machine Learning

## Installation

1. Cloner le dépot

```bash
git clone https://github.com/TristanLBD/Machine-Learning.git
```

2. Installer les dépendances python :

```bash
pip install -r requirements.txt
```

## Exécution de l'application

```bash
streamlit run main.py
```

## Dépendances principales

-   streamlit (Création d'apps web interactives pour le Machine Learning et la data)
-   pandas (Manipulation et analyse de données.)
-   numpy (Calculs numériques rapides, tableaux multidimensionnels)
-   matplotlib (Visualisation de données via des graphiques / courbes etc...)
-   seaborn (Visualisations statistiques avec un style amélioré)
-   scikit-learn (algorithmes de machine learning)
