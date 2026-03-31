import pandas as pd
import os
import numpy as np

# 1. Robust path setup so it runs perfectly from anywhere
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_path = os.path.join(script_dir, '..', 'data', 'raw', 'lending_club_data.csv')
ref_data_path = os.path.join(script_dir, '..', 'data', 'reference', 'reference.csv')
prod_batch_dir = os.path.join(script_dir, '..', 'data', 'production_batches')

# Ensure the output directories exist
os.makedirs(os.path.dirname(ref_data_path), exist_ok=True)
os.makedirs(prod_batch_dir, exist_ok=True)

print("Loading the massive dataset... this might take a minute.")
# We only load the columns we actually need to save memory and speed things up
columns_to_keep = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 
    'emp_length', 'home_ownership', 'annual_inc', 'purpose', 
    'dti', 'loan_status', 'issue_d'
]

# Load the raw data
df = pd.read_csv(raw_data_path, usecols=columns_to_keep)

print("Filtering for clear defaults and paid loans...")
# We only want loans that are definitely finished (either paid off or defaulted)
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]

# Convert the target variable to a clean binary: 0 for Paid, 1 for Default (Charged Off)
df['target'] = np.where(df['loan_status'] == 'Charged Off', 1, 0)
df = df.drop('loan_status', axis=1)

# Clean up some text columns into numbers (e.g., ' 36 months' -> 36)
df['term'] = df['term'].str.extract('(\d+)').astype(float)
df['emp_length'] = df['emp_length'].str.extract('(\d+)').astype(float)

# Drop missing values to keep our baseline model clean
df = df.dropna()

# Sort by issue date so we can simulate the passage of time accurately
df['issue_d'] = pd.to_datetime(df['issue_d'])
df = df.sort_values('issue_d')

print("Splitting into Reference (History) and Production (The Future)...")

# 1. The Reference Data (First 50,000 rows)
# This is what we will use to train our baseline model
reference_data = df.iloc[:50000]
reference_data.to_csv(ref_data_path, index=False)
print(f"Saved reference.csv with {len(reference_data)} rows.")

# 2. The Production Batches (The next 30,000 rows, split into 30 days)
# This simulates new loan applications arriving daily
production_data = df.iloc[50000:80000]

batch_size = 1000
for i in range(30):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    daily_batch = production_data.iloc[start_idx:end_idx]
    
    # Save each batch as a separate file (day_1.csv, day_2.csv, etc.)
    batch_filename = os.path.join(prod_batch_dir, f'day_{i+1}.csv')
    daily_batch.to_csv(batch_filename, index=False)

print("Successfully created 30 daily production batches!")
print("Data pipeline preparation complete.")