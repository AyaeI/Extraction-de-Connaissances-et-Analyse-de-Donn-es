import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import scipy.cluster.hierarchy as sch
from tabulate import tabulate

# Load the dataset
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 
           'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
data = pd.read_csv('adult/cleanadult.data', names=columns, header=0)

# Select all columns for clustering
data_clustering = data.copy()

# Encode categorical variables
categorical_columns =['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 
           'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
for col in categorical_columns:
    encoder = LabelEncoder()
    data_clustering[col] = encoder.fit_transform(data_clustering[col])

# Normalize the data
scaler = StandardScaler()
data_clustering[data_clustering.columns] = scaler.fit_transform(data_clustering[data_clustering.columns])

# Perform hierarchical clustering
print("Hierarchical Clustering")
cah = AgglomerativeClustering(n_clusters=5)
cah_clusters = cah.fit_predict(data_clustering)

# Ensure all cluster indices are covered using LabelEncoder
label_encoder_cah = LabelEncoder()
cah_clusters_encoded = label_encoder_cah.fit_transform(cah_clusters)

# Plot dendrogram for CAH
plt.figure(figsize=(10, 6))
plt.title('Dendrogram - CAH')
plt.ylabel('Distance')
dendrogram = sch.dendrogram(sch.linkage(data_clustering, method='ward'))
plt.show()


# K-Means Clustering Section

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns

# Perform k-means clustering
print("K-means Clustering")
kmeans = KMeans(n_clusters=5, max_iter=300, n_init=10, random_state=42)
kmeans_clusters = kmeans.fit_predict(data_clustering)

# Ensure all cluster indices are covered using LabelEncoder
label_encoder_kmeans = LabelEncoder()
kmeans_clusters_encoded = label_encoder_kmeans.fit_transform(kmeans_clusters)

# Add the cluster labels to the dataset
data_clustering['Cluster'] = kmeans_clusters_encoded

# Create a parallel coordinate plot
plt.figure(figsize=(10, 6))

# Reduce dimensionality with t-SNE
tsne = TSNE(n_components=2)
tsne_data = tsne.fit_transform(data_clustering.drop('income', axis=1))  # Drop 'income' if it's not numeric

# Using Seaborn's scatterplot
sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=kmeans_clusters_encoded, palette='viridis')
plt.title('t-SNE Visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()


# Comparison and Evaluation Section
print("Comparing Clustering Methods")

# Convert true labels to integers (assuming 'income' is the label)
true_labels = data['income'].apply(lambda x: 0 if x == ' <=50K' else 1)

# Function to calculate global statistics
def calculate_global_statistics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Function to calculate class-specific metrics
def calculate_class_metrics(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    sensitivity = np.divide(TP, (TP + FN), out=np.zeros_like(TP, dtype=float), where=(TP + FN)!=0)
    specificity = np.divide(TN, (TN + FP), out=np.zeros_like(TN, dtype=float), where=(TN + FP)!=0)
    detection_rate = TP / cm.sum()
    detection_prevalence = (TP + FP) / cm.sum()
    balanced_accuracy = (sensitivity + specificity) / 2

    metrics = {
        'Class': classes,  
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Detection Rate': detection_rate,
        'Detection Prevalence': detection_prevalence,
        'Balanced Accuracy': balanced_accuracy,
    }

    return pd.DataFrame(metrics)

# Combine confusion matrices
cah_cm_encoded = confusion_matrix(true_labels, cah_clusters_encoded)

# Plot CAH confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cah_cm_encoded, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('CAH Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Calculate and print metrics for CAH
print('\nGlobal Statistics - CAH:')
cah_stats_encoded = calculate_global_statistics(true_labels, cah_clusters_encoded)
print(tabulate(cah_stats_encoded.items(), headers=['Metric', 'Value'], tablefmt='psql'))

# Plot KMeans confusion matrix
kmeans_cm_encoded = confusion_matrix(true_labels, kmeans_clusters_encoded)
plt.figure(figsize=(8, 6))
sns.heatmap(kmeans_cm_encoded, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('KMeans Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Calculate and print metrics for KMeans
print('\nGlobal Statistics - KMeans:')
kmeans_stats_encoded = calculate_global_statistics(true_labels, kmeans_clusters_encoded)
print(tabulate(kmeans_stats_encoded.items(), headers=['Metric', 'Value'], tablefmt='psql'))


# Compare clusters from CAH and KMeans

# Plot comparison confusion matrix
comparison_cm = confusion_matrix(cah_clusters_encoded, kmeans_clusters_encoded)
plt.figure(figsize=(8, 6))
sns.heatmap(comparison_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix: CAH Clusters vs KMeans Clusters')
plt.xlabel('KMeans Clusters')
plt.ylabel('CAH Clusters')
plt.show()

# Calculate global statistics for comparison_cm
comparison_stats = calculate_global_statistics(cah_clusters_encoded, kmeans_clusters_encoded)
print('\nGlobal Statistics - CAH vs KMeans:')
print(tabulate(comparison_stats.items(), headers=['Metric', 'Value'], tablefmt='psql'))

# Calculate class-specific metrics for comparison_cm
comparison_metrics = calculate_class_metrics(cah_clusters_encoded, kmeans_clusters_encoded, label_encoder_cah.classes_)
print('\nClass-specific Metrics - CAH vs KMeans:')
print(tabulate(comparison_metrics, headers='keys', tablefmt='psql',  showindex=False))
