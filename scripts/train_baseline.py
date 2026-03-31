import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
ref_data_path = os.path.join(script_dir, '..', 'data', 'reference', 'reference.csv')
model_save_path = os.path.join(script_dir, '..', 'models', 'baseline_rf_model.joblib')
features_save_path = os.path.join(script_dir, '..', 'models', 'feature_columns.joblib')

print("Loading reference data...")
df = pd.read_csv(ref_data_path)

X = df.drop(['target', 'issue_d'], axis=1)

print("Encoding categorical variables...")
X = pd.get_dummies(X, drop_first=True)
y = df['target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the Baseline Random Forest Model... (This might take 10-20 seconds)")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_val)
acc = accuracy_score(y_val, predictions)
print(f"Baseline Model Accuracy: {acc * 100:.2f}%")

print("Saving the model for the automated pipeline...")
joblib.dump(model, model_save_path)
joblib.dump(list(X.columns), features_save_path)

print(f"Success! Model and feature map saved to the /models folder.")
