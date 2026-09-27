import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = "dataset/data"
MODEL_PATH = "model/isl_model.pkl"

X = []
y = []

# Read all CSV files
for file in os.listdir(DATA_PATH):

    if file.endswith(".csv"):

        label = os.path.splitext(file)[0]

        file_path = os.path.join(DATA_PATH, file)

        data = pd.read_csv(file_path, header=None)

        X.extend(data.values)
        y.extend([label] * len(data))

        print(f"Loaded {label}: {len(data)} samples")

X = np.array(X)
y = np.array(y)

print("\nTotal samples:", len(X))
print("Classes:", np.unique(y))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("\nTraining model...")

model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Save model
os.makedirs("model", exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print(MODEL_PATH)