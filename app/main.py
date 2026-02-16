import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Heart Disease Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "heart_disease_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "models", "model_features.pkl")

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def home():
    return {"message": "Heart Disease API is running!"}

@app.post("/predict")
def predict(data: PatientData):
    df = pd.DataFrame([data.model_dump()])
    
    df = df[features]
    
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return {
        "heart_disease_risk": int(prediction),
        "probability": round(float(probability), 2)
    }