import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import joblib
import os

# ================================
# 1. LOAD FEATURE DATA
# ================================

# Replace this path with your actual feature CSV
FEATURE_CSV = "data/training_features.csv"

if not os.path.exists(FEATURE_CSV):
    raise FileNotFoundError(
        "Feature CSV not found. Place training_features.csv inside data/ folder."
    )

df = pd.read_csv(FEATURE_CSV)

# ================================
# 2. SELECT NUMERIC FEATURES
# ================================

numeric_cols = df.select_dtypes(include=["number"]).columns

if len(numeric_cols) == 0:
    raise ValueError("No numeric feature columns detected.")

X = df[numeric_cols].fillna(0)

# ================================
# 3. FEATURE SCALING
# ================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# 4. DEFINE TARGET (EXPLORATORY)
# ================================

# Since you do not have labeled biological outcomes,
# we construct a composite exploratory target:

y = X_scaled.mean(axis=1)

# ================================
# 5. TRAIN MODEL
# ================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

model.fit(X_scaled, y)

# ================================
# 6. CROSS-VALIDATION CHECK
# ================================

cv_scores = cross_val_score(model, X_scaled, y, cv=3)

print("Cross-validation R2 scores:", cv_scores)
print("Mean CV R2:", np.mean(cv_scores))

# ================================
# 7. SAVE MODEL FILES
# ================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/eleprotect_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("Model and scaler saved successfully in models/ folder.")



