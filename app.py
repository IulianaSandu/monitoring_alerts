import os
import random
from datetime import datetime, timedelta
from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import IsolationForest
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
    
    # Preprocesare
    log_data['login_hour'] = pd.to_datetime(log_data['timestamp']).dt.hour
    log_data['login_day'] = pd.to_datetime(log_data['timestamp']).dt.day
    log_data['user_id'] = log_data['user_id'].astype('category').cat.codes
    log_data.drop_duplicates(inplace=True)

    # Caracteristici pentru model
    features = log_data[['user_id', 'login_hour', 'login_day']]  # Folosim mai multe caracteristici

    # Alegerea modelului Isolation Forest pentru detectarea anomaliilor
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)  # Ajustăm parametrii
    model.fit(features)
    log_data['anomaly'] = model.predict(features)

    anomalies = log_data[log_data['anomaly'] == -1]
    anomalies.to_csv('anomalies_detected.csv', index=False)
    
    # Evaluarea performanței modelului
    accuracy, f1, precision, recall, roc_auc = evaluate_model(log_data)
    print(f"Model Evaluation - Accuracy: {accuracy}, F1-Score: {f1}, Precision: {precision}, Recall: {recall}, ROC AUC: {roc_auc}")

    return anomalies

# Evaluarea performanței modelului folosind metrici de performanță
def evaluate_model(log_data):
    # Aici ar trebui să ai datele corecte pentru etichetele reale
    # În cazul nostru, simulăm etichetele reale ca fiind logările anormale
    y_true = [1 if i in range(950, 1000) else 0 for i in range(1000)]  # Simulăm etichete reale
    y_pred = log_data['anomaly'].apply(lambda x: 1 if x == -1 else 0).tolist()

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Interpretarea scorurilor
    print(f"Accuracy: {accuracy}")
    print(f"F1-Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"ROC AUC: {roc_auc}")
    print(f"Confusion Matrix:\nTN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    print("\nInterpretare scoruri:")
    print(f"Scorul de Acuratețe (Accuracy) este {accuracy*100:.2f}%, ceea ce înseamnă că aproximativ {accuracy*100:.2f}% din predicții sunt corecte.")
    print(f"F1-Score este {f1:.2f}, un indicator care arată echilibrul între precizie și recall. Un F1-Score mai mare sugerează un model echilibrat.")
    print(f"Precizia este {precision:.2f}, ceea ce indică procentul de predicții pozitive corecte (anomalii detectate corect).")
    print(f"Recall este {recall:.2f}, care arată capacitatea modelului de a detecta toate anomaliile. Un recall mai mare înseamnă că mai puține anomalii au fost ratate.")
    print(f"Scorul ROC AUC este {roc_auc:.2f}, ceea ce reflectă performanța generală a modelului într-un mod robust la toate pragurile de decizie.")

    return accuracy, f1, precision, recall, roc_auc

# Trimiterea unui email de alertă
def send_alert_email(anomalies):
    if anomalies.empty:
        return
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL
        msg['To'] = RECIPIENT
        msg['Subject'] = "Alerte de securitate - Logări neobișnuite detectate"
        
        body = f"""
        <h3>A fost detectat un număr de {len(anomalies)} logări anormale:</h3>
        <table border="1" style="border-collapse: collapse;">
            <tr>
                <th>User</th>
                <th>Timestamp</th>
                <th>Device</th>
                <th>Location</th>
            </tr>
        """
        
        for _, row in anomalies.iterrows():
            body += f"""
            <tr>
                <td>{row['user_id']}</td>
                <td>{row['timestamp']}</td>
                <td>{row['device']}</td>
                <td>{row['location']}</td>
            </tr>
            """
        body += "</table>"
        
        msg.attach(MIMEText(body, 'html'))
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL, EMAIL_PASSWORD)
            server.sendmail(EMAIL, RECIPIENT, msg.as_string())
    except Exception as e:
        print(f"Eroare la trimiterea emailului: {e}")

# Încărcarea alertelor pentru dashboard
def load_anomalies():
    if os.path.exists("anomalies_detected.csv"):
        return pd.read_csv("anomalies_detected.csv").to_dict(orient="records")
    return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/project')
def project():
    return render_template('project.html')

@app.route('/student')
def student():
    return render_template('student.html')

@app.route('/alerts')
def alerts():
    anomalies = load_anomalies()
    # Paginare sau căutare
    page = request.args.get('page', 1, type=int)
    items_per_page = 10
    paginated_anomalies = anomalies[(page - 1) * items_per_page: page * items_per_page]
    return render_template('alerts.html', anomalies=paginated_anomalies)

if __name__ == '__main__':
    anomalies = preprocess_and_detect()
    send_alert_email(anomalies)
    app.run(debug=True)
