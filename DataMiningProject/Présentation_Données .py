import pandas as pd
from tabulate import tabulate

# Charger le dataset
df = pd.read_csv('adult/adult.data')

# Renommer les colonnes
df.columns = ['age', 'workclass', 'flwgt', 'education', 'education_num', 'marital_status',
              'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
              'hours_per_week', 'native_country', 'income']

# Présentation des données
print("Premières lignes du dataset :")
print(tabulate(df.head(), headers='keys', tablefmt='pretty'))

print("\nInformations sur le dataset :")
print(df.info())

print("\nStatistiques descriptives :")
print(tabulate(df.describe(), headers='keys', tablefmt='pretty'))

# Préparation des données
df.dropna(inplace=True)  # Suppression des valeurs manquantes
df.drop_duplicates(inplace=True)  # Suppression des valeurs en double

print("\nDonnées après nettoyage :")
print(tabulate(df.head(), headers='keys', tablefmt='pretty'))

# Sauvegarder le dataset nettoyé
df.to_csv('adult/cleanadult.data', index=False)

print("\nLe dataset nettoyé a été sauvegardé dans le fichier 'cleanadult.data'.")
