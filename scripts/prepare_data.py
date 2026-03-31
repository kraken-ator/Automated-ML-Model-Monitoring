import pandas as pd
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_path = os.path.join(script_dir, '..', 'data', 'raw', 'lending_club_data.csv')
ref_data_path = os.path.join(script_dir, '..', 'data', 'reference', 'reference.csv')
prod_batch_dir = os.path.join(script_dir, '..', 'data', 'production_batches')

os.makedirs(os.path.dirname(ref_data_path), exist_ok=True)
os.makedirs(prod_batch_dir, exist_ok=True)

print("Loading the massive dataset... this might take a minute.")
columns_to_keep = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 
    'emp_length', 'home_ownership', 'annual_inc', 'purpose', 
    'dti', 'loan_status', 'issue_d'
]

df = pd.read_csv(raw_data_path, usecols=columns_to_keep)

print("Filtering for clear defaults and paid loans...")
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]

df['target'] = np.where(df['loan_status'] == 'Charged Off', 1, 0)
df = df.drop('loan_status', axis=1)

df['term'] = df['term'].str.extract('(\d+)').astype(float)
df['emp_length'] = df['emp_length'].str.extract('(\d+)').astype(float)

df = df.dropna()

df['issue_d'] = pd.to_datetime(df['issue_d'])
df = df.sort_values('issue_d')

print("Splitting into Reference (History) and Production (The Future)...")

reference_data = df.iloc[:50000]
reference_data.to_csv(ref_data_path, index=False)
print(f"Saved reference.csv with {len(reference_data)} rows.")

production_data = df.iloc[50000:80000]

batch_size = 1000
for i in range(30):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    daily_batch = production_data.iloc[start_idx:end_idx]
    
    batch_filename = os.path.join(prod_batch_dir, f'day_{i+1}.csv')
    daily_batch.to_csv(batch_filename, index=False)

print("Successfully created 30 daily production batches!")
print("Data pipeline preparation complete.")
