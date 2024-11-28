import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Charger le dataset
df = pd.read_csv('adult/cleanadult.data')

# Renommer les colonnes
df.columns = ['age', 'workclass', 'flwgt', 'education', 'education_num', 'marital_status',
              'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
              'hours_per_week', 'native_country', 'income']

# Fonction pour l'analyse de l'arbre de décision
def decision_tree_analysis(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy_dt = clf.score(X_test, y_test)
    print(f"Exactitude de l'arbre de decision : {accuracy_dt:.2f}")
    print(classification_report(y_test, y_pred))
    plt.figure(figsize=(20, 10))
    plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_, max_depth=3, fontsize=8)
    plt.show()
    return accuracy_dt

# Fonction pour l'analyse du modèle k-NN
def knn_analysis(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred)
    print(f"Exactitude de k-NN : {accuracy_knn:.2f}")
    print(classification_report(y_test, y_pred))
    return accuracy_knn

# Exécution des analyses
if __name__ == "__main__":
    # Préparation des caractéristiques et de la cible pour les modèles
    X = df.drop('income', axis=1)  # Caractéristiques
    y = df['income']  # Cible
    
    # Conversion des variables catégorielles en variables fictives
    X = pd.get_dummies(X, drop_first=True)

    # Analyse de l'arbre de décision
    print("Analyse de l'arbre de decision :")
    accuracy_dt = decision_tree_analysis(X, y)
    
    # Analyse de k-NN
    print("\nAnalyse de k-NN :")
    accuracy_knn = knn_analysis(X, y)
    
    # Comparaison et tracé
    plt.figure(figsize=(10, 6))
    methods = ['Decision Tree', 'k-NN']
    accuracies = [accuracy_dt, accuracy_knn]
    plt.bar(methods, accuracies, color=['blue', 'green'])
    plt.ylim(0.75, 0.88)
    plt.ylabel('Exactitude')
    plt.title('Comparaison des exactitudes entre Decision Tree et k-NN')
    plt.show()
