from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Heart Disease Prediction API")
    
model = joblib.load("notebooks/heart_disease_model.pkl")
features = joblib.load("notebooks/model_features.pkl")

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
    return{"message": "Heart Disease API is running!"}

@app.post("/predict")
def predict(data: PatientData):
    df = pd.DataFrame([data.model_dump()])
    
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return{
        "heart_disease_risk": int(prediction),
        "probability": round(float(probability), 2)
    }