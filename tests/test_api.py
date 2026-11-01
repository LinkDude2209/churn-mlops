
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model"] == "Random Forest"


def test_prediction_endpoint():

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in ["Churn", "No Churn"]

    assert 0 <= result["churn_probability"] <= 1

    assert 0 <= result["no_churn_probability"] <= 1

    assert abs(
        result["churn_probability"]
        + result["no_churn_probability"] - 1
    ) < 0.001


def test_prediction_rejects_invalid_input():
    response = client.post("/predict", json={})

    assert response.status_code == 422
