import pandas as pd
import joblib
import sqlite3
import os
from datetime import datetime, timedelta
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
ref_data_path = os.path.join(script_dir, '..', 'data', 'reference', 'reference.csv')
prod_batch_dir = os.path.join(script_dir, '..', 'data', 'production_batches')
model_path = os.path.join(script_dir, '..', 'models', 'baseline_rf_model.joblib')
features_path = os.path.join(script_dir, '..', 'models', 'feature_columns.joblib')
db_path = os.path.join(script_dir, '..', 'logs', 'drift_metrics.db')

print("Loading Reference Data and Baseline Model...")
reference_data = pd.read_csv(ref_data_path)
model = joblib.load(model_path)
expected_features = joblib.load(features_path)

ref_features = pd.get_dummies(reference_data.drop(['target', 'issue_d'], axis=1), drop_first=True)
for col in expected_features:
    if col not in ref_features.columns:
        ref_features[col] = 0
ref_features = ref_features[expected_features]

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

start_date = datetime(2026, 4, 1)

print("\n--- Starting 30-Day Automated Monitoring Simulation ---\n")

for i in range(1, 31):
    batch_name = f'day_{i}.csv'
    batch_path = os.path.join(prod_batch_dir, batch_name)
    current_date = (start_date + timedelta(days=i-1)).strftime('%Y-%m-%d')
    
    print(f"Processing {batch_name} (Date: {current_date})...")
    
    current_data = pd.read_csv(batch_path)
    
    current_data_features = pd.get_dummies(current_data.drop(['target', 'issue_d'], axis=1), drop_first=True)
    for col in expected_features:
        if col not in current_data_features.columns:
            current_data_features[col] = 0
    current_data_features = current_data_features[expected_features]
    
    predictions = model.predict(current_data_features)
    accuracy = accuracy_score(current_data['target'], predictions)
    
    drifted_features = 0
    numerical_cols = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti']
    
    for col in numerical_cols:
        stat, p_value = ks_2samp(reference_data[col], current_data[col])
        if p_value < 0.05:
            drifted_features += 1
            
    drift_detected = "YES" if drifted_features > 0 else "NO"
    dataset_size = len(current_data)
    action_req = "RETRAIN MODEL" if accuracy < 0.75 or drifted_features >= 3 else "ALL CLEAR"
    
    cursor.execute('''
        INSERT INTO daily_drift_logs 
        (run_date, batch_name, dataset_size, overall_data_drift, drifted_features_count, model_accuracy, action_required)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (current_date, batch_name, dataset_size, drift_detected, drifted_features, accuracy, action_req))
    
    conn.commit()

conn.close()
print("\n--- Simulation Complete! All 30 days logged to SQLite Database. ---")
