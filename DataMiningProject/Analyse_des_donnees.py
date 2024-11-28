import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset
df = pd.read_csv('adult/cleanadult.data')

# Renommer les colonnes
df.columns = ['age', 'workclass', 'flwgt', 'education', 'education_num', 'marital_status',
              'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
              'hours_per_week', 'native_country', 'income']

# Univariate analysis
def univariate_analysis(X):
    for column in X.columns:
        plt.figure(figsize=(10, 6))
        
        # Check if the column is categorical or numerical
        if X[column].dtype == 'object':
            X[column].value_counts().plot(kind='bar')
            plt.xticks(ticks=range(len(X[column].value_counts())), labels=range(1, len(X[column].value_counts())+1))
        else:
            X[column].plot(kind='hist', bins=30, edgecolor='black', alpha=0.7)
        
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()


# Bivariate analysis
# Bivariate analysis
def bivariate_analysis(X):
    selected_features = X.columns[:5]  
    for i in range(len(selected_features)):
        for j in range(i+1, len(selected_features)):
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=X[selected_features[i]], y=X[selected_features[j]])
            if X[selected_features[i]].dtype == 'object':
                plt.xticks(ticks=range(len(X[selected_features[i]].value_counts())), labels=range(1, len(X[selected_features[i]].value_counts())+1))
            if X[selected_features[j]].dtype == 'object':
                plt.yticks(ticks=range(len(X[selected_features[j]].value_counts())), labels=range(1, len(X[selected_features[j]].value_counts())+1))
                
            plt.title(f'Scatter plot between {selected_features[i]} and {selected_features[j]}')
            plt.xlabel(selected_features[i])
            plt.ylabel(selected_features[j])
            plt.show()


# Exécution des analyses
if __name__ == "__main__":

    # Appel des fonctions pour l'analyse univariée et bivariée
    print("Analyse univariée :")
    univariate_analysis(df)

    print("Analyse bivariée :")
    bivariate_analysis(df)
    
   