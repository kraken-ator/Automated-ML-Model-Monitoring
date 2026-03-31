import sqlite3
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, '..', 'logs', 'drift_metrics.db')
csv_path = os.path.join(script_dir, '..', 'logs', 'tableau_dashboard_data.csv')

print("Connecting to the monitoring database...")
conn = sqlite3.connect(db_path)

df = pd.read_sql_query("SELECT * FROM daily_drift_logs", conn)

df.to_csv(csv_path, index=False)
conn.close()

print(f"Success! Data exported to: {csv_path}")
print("You are ready for Tableau.")
