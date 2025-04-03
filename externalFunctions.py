import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def getDataframe():
    df_2015 = pd.read_csv('datas/F1_2015_2016.csv', parse_dates=['Date'],dayfirst=True)
    df_2016 = pd.read_csv('datas/F1_2016_2017.csv', parse_dates=['Date'],dayfirst=True)
    df_2017 = pd.read_csv('datas/F1_2017_2018.csv', parse_dates=['Date'],dayfirst=True)
    df_2018 = pd.read_csv('datas/F1_2018_2019.csv', parse_dates=['Date'],dayfirst=True)
    df_2019 = pd.read_csv('datas/F1_2019_2020.csv', parse_dates=['Date'],dayfirst=True)
    df_2020 = pd.read_csv('datas/F1_2020_2021.csv', parse_dates=['Date'],dayfirst=True)
    df_2021 = pd.read_csv('datas/F1_2021_2022.csv', parse_dates=['Date'],dayfirst=True)
    df_2015['Saison'] = '2015'
    df_2016['Saison'] = '2016'
    df_2017['Saison'] = '2017'
    df_2018['Saison'] = '2018'
    df_2019['Saison'] = '2019'
    df_2020['Saison'] = '2020'
    df_2021['Saison'] = '2021'
    df_2015=df_2015.dropna(subset = ['HomeTeam'], axis=0)
    df = pd.concat([df_2015, df_2016, df_2017, df_2018, df_2019, df_2020, df_2021])
    df = df.dropna(subset=['HTHG'])

    return df


def retirerColonnes(df):
    colonnes_utiles = [
        'Saison', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
        'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY',
        'B365H', 'B365D', 'B365A'
    ]
    return df[colonnes_utiles]

def renameDataframe(df):
    df.rename(columns={
        'Date': 'Date_Match',
        'HomeTeam': 'Equipe_Domicile',
        'AwayTeam': 'Equipe_Exterieure',
        'FTHG': 'Buts_Domicile',
        'FTAG': 'Buts_Exterieur',
        'FTR': 'Resultat',
        'HTHG': 'Buts_Domicile_Mi_Temps',
        'HTAG': 'Buts_Exterieur_Mi_Temps',
        'HS': 'Tirs_Domicile',
        'AS': 'Tirs_Exterieur',
        'HST': 'Tirs_Cadres_Domicile',
        'AST': 'Tirs_Cadres_Exterieur',
        'HC': 'Corners_Domicile',
        'AC': 'Corners_Exterieur',
        'HF': 'Fautes_Domicile',
        'AF': 'Fautes_Exterieur',
        'HY': 'Cartons_Jaunes_Domicile',
        'AY': 'Cartons_Jaunes_Exterieur'
    }, inplace=True)
    return df

def rolling_averages(group, cols, new_cols, window=5):
    group = group.sort_values("Date_Match")
    rolling_stats = group[cols].rolling(window, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Victoire Domicile', 'Victoire Extérieure', 'Match Nul'],
                yticklabels=['Victoire Domicile', 'Victoire Extérieure', 'Match Nul'])
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités')
    plt.title('Matrice de Confusion')
    plt.savefig('matrice_confusion.png')
    plt.close()

def plot_feature_importances(feature_importance_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.sort_values(by='Importance', ascending=False), palette='viridis')
    plt.title('Importances des Caractéristiques')
    plt.xlabel('Importance')
    plt.ylabel('Caractéristiques')
    plt.savefig('feature_importances.png')
    plt.close()

