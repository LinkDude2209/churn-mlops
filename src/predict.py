
import os
import joblib
import pandas as pd


MODEL_PATH = "models/model.pkl"
DATA_PATH = "data/churn.csv"


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run python src/train.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_customer(customer_data):
    model = load_model()

    # Convert input dictionary into a DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Get prediction and probability
    prediction = int(model.predict(customer_df)[0])

    probabilities = model.predict_proba(customer_df)[0]

    churn_probability = float(probabilities[1])

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(churn_probability, 4),
        "confidence": round(
            float(max(probabilities)), 4
        )
    }


if __name__ == "__main__":

    # Load a sample customer from our dataset
    df = pd.read_csv(DATA_PATH)

    sample = df.iloc[0].copy()

    actual = sample["Churn"]

    sample = sample.drop(
        ["customerID", "Churn"]
    )

    sample["TotalCharges"] = pd.to_numeric(
        sample["TotalCharges"],
        errors="coerce"
    )

    result = predict_customer(sample.to_dict())

    print("\nCUSTOMER CHURN PREDICTION")
    print("------------------------")

    print("Prediction:", result["prediction"])

    print(
        "Churn probability:",
        result["churn_probability"] * 100,
        "%"
    )

    print(
        "Prediction confidence:",
        result["confidence"] * 100,
        "%"
    )

    print("Actual outcome:", actual)
