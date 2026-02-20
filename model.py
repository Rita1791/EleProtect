import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_model(feature_csv_path):
    df = pd.read_csv(feature_csv_path)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    X = df[numeric_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y = X_scaled.mean(axis=1)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/eleprotect_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    return "Model trained successfully."

def predict_score(feature_df):
    model = joblib.load("models/eleprotect_model.joblib")
    scaler = joblib.load("models/scaler.joblib")

    numeric_cols = feature_df.select_dtypes(include=["number"]).columns
    X = feature_df[numeric_cols].fillna(0)

    X_scaled = scaler.transform(X)
    scores = model.predict(X_scaled)

    feature_df["ML_Score"] = scores
    return feature_df.sort_values("ML_Score", ascending=False)
