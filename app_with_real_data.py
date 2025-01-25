import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Calea către fișierul de date
dataset_path = 'creditcard.csv'

def load_data():
    """Încarcă datele din fișierul CSV (folosind doar un subset pentru testare rapidă)"""
    print("Încărcare date...")
    df = pd.read_csv(dataset_path, nrows=100000)  # Folosim doar un subset pentru testare rapidă
    return df

def preprocess_data(df):
    """Preprocesarea datelor pentru a le transforma într-un format utilizabil de modele"""
    features = df.drop(['Time', 'Class'], axis=1)  # Eliminăm coloanele Time și Class
    labels = df['Class']  # Coloana 'Class' este eticheta (frauda: 1, normal: 0)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)  # Standardizăm datele
    return features_scaled, labels

def detect_anomalies(features_scaled):
    """Detectarea anomaliilor folosind 3 modele diferite"""
    print("Detectare anomalii...")

    # Modele de detectare a anomaliilor
    isolation_forest = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
    one_class_svm = OneClassSVM(nu=0.01, kernel='rbf', gamma=0.05)  # Ajustăm gamma și nu pentru o rulare mai rapidă
    lof_model = LocalOutlierFactor(n_neighbors=30, contamination=0.05)  # Creștem numărul de vecini pentru LOF

    # Prezicerea anomaliilor
    isolation_forest_pred = isolation_forest.fit_predict(features_scaled)
    one_class_svm_pred = one_class_svm.fit_predict(features_scaled)
    lof_pred = lof_model.fit_predict(features_scaled)

    return isolation_forest_pred, one_class_svm_pred, lof_pred

def evaluate_performance(y_true, y_pred):
    """Calculează metricile de performanță pentru a evalua modelele"""
    # Convertește predicțiile de la -1, 1 la 1 (anomalii) și 0 (normale)
    y_pred = [1 if x == -1 else 0 for x in y_pred]  # -1 -> 1 (anomalii), 1 -> 0 (normale)
    
    # Calculăm metricile de performanță
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_pred)

    return accuracy, f1, precision, recall, roc_auc

def main():
    # Încărcăm și preprocesăm datele
    df = load_data()
    features_scaled, labels = preprocess_data(df)

    # Aplicăm modelele pentru detectarea anomaliilor
    isolation_forest_pred, one_class_svm_pred, lof_pred = detect_anomalies(features_scaled)

    # Evaluăm performanța fiecărui model
    isolation_forest_metrics = evaluate_performance(labels, isolation_forest_pred)
    one_class_svm_metrics = evaluate_performance(labels, one_class_svm_pred)
    lof_metrics = evaluate_performance(labels, lof_pred)

    # Afișăm rezultatele
    print("Performanță Isolation Forest:")
    print(f"Acuratețe: {isolation_forest_metrics[0]}, F1: {isolation_forest_metrics[1]}, Precizie: {isolation_forest_metrics[2]}, Recall: {isolation_forest_metrics[3]}, AUC-ROC: {isolation_forest_metrics[4]}")
    
    print("\nPerformanță One-Class SVM:")
    print(f"Acuratețe: {one_class_svm_metrics[0]}, F1: {one_class_svm_metrics[1]}, Precizie: {one_class_svm_metrics[2]}, Recall: {one_class_svm_metrics[3]}, AUC-ROC: {one_class_svm_metrics[4]}")
    
    print("\nPerformanță LOF:")
    print(f"Acuratețe: {lof_metrics[0]}, F1: {lof_metrics[1]}, Precizie: {lof_metrics[2]}, Recall: {lof_metrics[3]}, AUC-ROC: {lof_metrics[4]}")

if __name__ == '__main__':
    main()
