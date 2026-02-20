import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Dummy training dataset (since you do not have labeled outcomes)
data = {
    "mutation_frequency": [0.8, 0.7, 0.9, 0.85, 0.75, 0.88],
    "conservation_score": [1, 1, 1, 0.9, 1, 0.95],
    "retrogene_variability": [0.1, 0.2, 0.05, 0.3, 0.15, 0.08]
}

df = pd.DataFrame(data)

X = df
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = X_scaled.mean(axis=1)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_scaled, y)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/eleprotect_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("Model files created.")
