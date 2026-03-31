import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Robust path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
ref_data_path = os.path.join(script_dir, '..', 'data', 'reference', 'reference.csv')
model_save_path = os.path.join(script_dir, '..', 'models', 'baseline_rf_model.joblib')
features_save_path = os.path.join(script_dir, '..', 'models', 'feature_columns.joblib')

print("Loading reference data...")
df = pd.read_csv(ref_data_path)

# 2. Define Features (X) and Target (y)
# We drop 'issue_d' because a model shouldn't train on the timestamp itself
X = df.drop(['target', 'issue_d'], axis=1)

# Handle categorical variables (like 'purpose' or 'home_ownership') using One-Hot Encoding
print("Encoding categorical variables...")
X = pd.get_dummies(X, drop_first=True)
y = df['target']

# 3. Split into train and validation sets to verify our baseline works
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Random Forest Model
print("Training the Baseline Random Forest Model... (This might take 10-20 seconds)")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 5. Quick Evaluation
predictions = model.predict(X_val)
acc = accuracy_score(y_val, predictions)
print(f"Baseline Model Accuracy: {acc * 100:.2f}%")

# 6. Save the Model and the exact feature columns
print("Saving the model for the automated pipeline...")
joblib.dump(model, model_save_path)
joblib.dump(list(X.columns), features_save_path)

print(f"Success! Model and feature map saved to the /models folder.")