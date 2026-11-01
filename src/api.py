
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ----------------------------------
# 1. LOAD TRAINED MODEL
# ----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run python src/train.py first."
    )

model = joblib.load(MODEL_PATH)


# ----------------------------------
# 2. INITIALIZE FASTAPI
# ----------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML-powered telecom customer churn prediction service",
    version="1.0.0"
)


# ----------------------------------
# 3. DEFINE INPUT SCHEMA
# ----------------------------------

class CustomerData(BaseModel):

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)

    PhoneService: str
    MultipleLines: str
    InternetService: str

    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str

    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


# ----------------------------------
# 4. HEALTH CHECK
# ----------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ----------------------------------
# 5. MODEL INFORMATION
# ----------------------------------

@app.get("/model-info")
def model_info():

    return {
        "model": "Random Forest",
        "problem": "Binary Classification",
        "target": "Customer Churn",
        "model_file": "models/model.pkl"
    }


# ----------------------------------
# 6. PREDICTION ENDPOINT
# ----------------------------------

@app.post("/predict")
def predict(data: CustomerData):

    try:

        customer = pd.DataFrame([
            data.model_dump()
        ])

        prediction = int(model.predict(customer)[0])

        probabilities = model.predict_proba(customer)[0]

        churn_probability = float(probabilities[1])
        no_churn_probability = float(probabilities[0])

        return {
            "prediction": (
                "Churn" if prediction == 1 else "No Churn"
            ),
            "churn_probability": round(
                churn_probability, 4
            ),
            "no_churn_probability": round(
                no_churn_probability, 4
            ),
            "model": "Random Forest"
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
