import sqlite3

conn = sqlite3.connect('../logs/drift_metrics.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS daily_drift_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date DATE,
    batch_name TEXT,
    dataset_size INTEGER,
    overall_data_drift TEXT,
    drifted_features_count INTEGER,
    model_accuracy REAL,
    action_required TEXT
)
''')

conn.commit()
conn.close()

print("Database and schema created successfully in the logs folder!")
