import os
import random
from datetime import datetime, timedelta
from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Încarcă variabilele de configurare din fișierul .env
load_dotenv()

SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
EMAIL = os.getenv("EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
RECIPIENT = os.getenv("RECIPIENT")

app = Flask(__name__)

# Generarea datelor fictive de autentificare
def generate_fake_data():
    users = [f"user_{i}" for i in range(1, 21)]  # 20 utilizatori fictivi
    devices = ['PC', 'Laptop', 'Mobile']
    locations = ['București', 'Cluj', 'Timișoara', 'Iași', 'Brașov']
    data = []
    
    # Generăm logări anormale: utilizatori care se loghează din locații noi
    anomalies = random.sample(users, 5)  # 5 utilizatori vor avea logări anormale
    
    for _ in range(1000):  # 1000 de logări fictive
        user = random.choice(users)
        timestamp = datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        device = random.choice(devices)
        location = random.choice(locations)
        
        # Creăm logări anormale pentru utilizatorii selectați
        if user in anomalies:
            location = random.choice([loc for loc in locations if loc != 'București'])  # schimbăm locația
            timestamp -= timedelta(hours=random.randint(12, 24))  # logare într-un interval de timp neobișnuit
        
        data.append([user, timestamp.strftime('%Y-%m-%d %H:%M:%S'), device, location])
    
    return pd.DataFrame(data, columns=['user_id', 'timestamp', 'device', 'location'])

# Preprocesarea datelor și detectarea anomaliilor
def preprocess_and_detect():
    log_data = generate_fake_data()
    log_data.to_csv('loguri.csv', index=False)
    
    # Crearea coloanelor 'login_hour' și 'login_day' din 'timestamp'
    log_data['timestamp'] = pd.to_datetime(log_data['timestamp'])
    log_data['login_hour'] = log_data['timestamp'].dt.hour
    log_data['login_day'] = log_data['timestamp'].dt.day

    # Transformarea user_id în valoare numerică
    log_data['user_id'] = log_data['user_id'].astype('category').cat.codes

    # Preprocesare: Normalizare/standardizare a datelor
    scaler = StandardScaler()
    features = scaler.fit_transform(log_data[['user_id', 'login_hour', 'login_day']])

    # Aplicarea modelului Isolation Forest cu hiperparametri ajustați
    isolation_forest_model = IsolationForest(n_estimators=200, contamination=0.05, max_samples='auto', random_state=42)
    isolation_forest_model.fit(features)
    log_data['anomaly_isolation_forest'] = isolation_forest_model.predict(features)

    # Aplicarea modelului One-Class SVM cu kernel RBF (fără random_state)
    one_class_svm_model = OneClassSVM(nu=0.05, kernel='rbf', gamma=0.1)
    one_class_svm_model.fit(features)
    log_data['anomaly_one_class_svm'] = one_class_svm_model.predict(features)

    # Aplicarea modelului LOF (Local Outlier Factor) cu n_neighbors mai mare
    lof_model = LocalOutlierFactor(n_neighbors=30, contamination=0.05)
    lof_anomalies = lof_model.fit_predict(features)
    log_data['anomaly_lof'] = lof_anomalies

    # Salvarea anomaliilor în fișiere CSV
    isolation_forest_anomalies = log_data[log_data['anomaly_isolation_forest'] == -1]
    isolation_forest_anomalies.to_csv('anomalies_isolation_forest.csv', index=False)

    one_class_svm_anomalies = log_data[log_data['anomaly_one_class_svm'] == -1]
    one_class_svm_anomalies.to_csv('anomalies_one_class_svm.csv', index=False)

    lof_anomalies_df = log_data[log_data['anomaly_lof'] == -1]
    lof_anomalies_df.to_csv('anomalies_lof.csv', index=False)

    # Evaluarea performanței pentru fiecare model
    isolation_forest_metrics = evaluate_model(log_data, 'isolation_forest')
    one_class_svm_metrics = evaluate_model(log_data, 'one_class_svm')
    lof_metrics = evaluate_model(log_data, 'lof')

    return isolation_forest_anomalies, one_class_svm_anomalies, lof_anomalies_df, isolation_forest_metrics, one_class_svm_metrics, lof_metrics

# Evaluarea performanței modelului folosind metrici de performanță
def evaluate_model(log_data, model_name):
    y_true = [1 if i in range(950, 1000) else 0 for i in range(1000)]  # Simulăm etichete reale
    y_pred = log_data[f'anomaly_{model_name}'].apply(lambda x: 1 if x == -1 else 0).tolist()

    # Verifică dacă dimensiunile y_true și y_pred sunt corecte
    if len(y_true) != len(y_pred):
        print(f"Warning: y_true and y_pred have different lengths. y_true: {len(y_true)}, y_pred: {len(y_pred)}")
        # Asigură-te că predicțiile au aceeași lungime ca etichetele reale
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return accuracy, f1, precision, recall, roc_auc

# Trimiterea unui email de alertă
def send_alert_email(isolation_forest_anomalies, one_class_svm_anomalies, lof_anomalies_df):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL
        msg['To'] = RECIPIENT
        msg['Subject'] = "Alerte de Securitate - Logări Neobișnuite Detectate"
        
        # Mesajul de alertă
        body = f"""
        <h3>Raport privind anomaliile detectate:</h3>
        <p><b>Isolation Forest:</b> {'A fost detectată(n) ' + str(len(isolation_forest_anomalies)) + ' anomalie(ie).' if not isolation_forest_anomalies.empty else 'Nu au fost detectate anomalii.'}</p>
        <p><b>One-Class SVM:</b> {'A fost detectată(n) ' + str(len(one_class_svm_anomalies)) + ' anomalie(ie).' if not one_class_svm_anomalies.empty else 'Nu au fost detectate anomalii.'}</p>
        <p><b>LOF (Local Outlier Factor):</b> {'A fost detectată(n) ' + str(len(lof_anomalies_df)) + ' anomalie(ie).' if not lof_anomalies_df.empty else 'Nu au fost detectate anomalii.'}</p>
        """
        
        msg.attach(MIMEText(body, 'html'))

        # Conectarea și trimiterea emailului
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL, EMAIL_PASSWORD)
            server.sendmail(EMAIL, RECIPIENT, msg.as_string())
        print("Email trimis cu succes!")

    except Exception as e:
        print(f"Eroare la trimiterea emailului: {e}")

# Încărcarea alertelor pentru dashboard
def load_anomalies():
    if os.path.exists("anomalies_isolation_forest.csv"):
        isolation_forest = pd.read_csv("anomalies_isolation_forest.csv")
    else:
        isolation_forest = pd.DataFrame()

    if os.path.exists("anomalies_one_class_svm.csv"):
        one_class_svm = pd.read_csv("anomalies_one_class_svm.csv")
    else:
        one_class_svm = pd.DataFrame()

    if os.path.exists("anomalies_lof.csv"):
        lof = pd.read_csv("anomalies_lof.csv")
    else:
        lof = pd.DataFrame()

    return isolation_forest, one_class_svm, lof
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/project')
def project():
    return render_template('project.html')

@app.route('/student')
def student():
    return render_template('student.html')
# Pagina de alertă
@app.route('/alerts')
def alerts():
    isolation_forest_anomalies, one_class_svm_anomalies, lof_anomalies = load_anomalies()
    return render_template('alerts.html', 
                           isolation_forest_anomalies=isolation_forest_anomalies.to_dict(orient="records"),
                           one_class_svm_anomalies=one_class_svm_anomalies.to_dict(orient="records"),
                           lof_anomalies=lof_anomalies.to_dict(orient="records"))
@app.route('/results')
def rezultate():
    isolation_forest_anomalies, one_class_svm_anomalies, lof_anomalies_df, \
        isolation_forest_metrics, one_class_svm_metrics, lof_metrics = preprocess_and_detect()
    
    return render_template('results.html', 
                           isolation_forest_metrics=isolation_forest_metrics,
                           one_class_svm_metrics=one_class_svm_metrics,
                           lof_metrics=lof_metrics)


if __name__ == '__main__':
    # Preprocesare și detectare
    isolation_forest_anomalies, one_class_svm_anomalies, lof_anomalies_df, \
        isolation_forest_metrics, one_class_svm_metrics, lof_metrics = preprocess_and_detect()
    
    # Dacă există anomalii, trimite un email
    send_alert_email(isolation_forest_anomalies, one_class_svm_anomalies, lof_anomalies_df)

    # Rulare aplicație Flask
    app.run(debug=False)
