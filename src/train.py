
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ----------------------------------
# 1. CONFIGURATION
# ----------------------------------

DATA_PATH = "data/churn.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42


# ----------------------------------
# 2. LOAD DATASET
# ----------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ----------------------------------
# 3. DATA CLEANING
# ----------------------------------

# Convert TotalCharges to numeric.
# Invalid or empty values become NaN.

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove customer ID because it is not predictive.

df.drop(columns=["customerID"], inplace=True)

# Convert target labels to binary.

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

df.dropna(subset=["Churn"], inplace=True)

X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)


# ----------------------------------
# 4. IDENTIFY FEATURE TYPES
# ----------------------------------

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)


# ----------------------------------
# 5. PREPROCESSING PIPELINE
# ----------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ----------------------------------
# 6. TRAIN / TEST SPLIT
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ----------------------------------
# 7. DEFINE MODELS
# ----------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# ----------------------------------
# 8. TRAIN AND EVALUATE
# ----------------------------------

results = {}
trained_models = {}

for name, classifier in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(
            y_test, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_test, predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(
            y_test, probabilities
        ))
    }

    results[name] = metrics
    trained_models[name] = pipeline

    print("\nResults:", name)
    print(json.dumps(metrics, indent=4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# ----------------------------------
# 9. SELECT BEST MODEL
# ----------------------------------

best_model_name = max(
    results,
    key=lambda name: results[name]["f1_score"]
)

best_model = trained_models[best_model_name]

print("\nBest model:", best_model_name)


# ----------------------------------
# 10. SAVE MODEL
# ----------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "model.pkl"
)

joblib.dump(best_model, model_path)


# ----------------------------------
# 11. SAVE METRICS
# ----------------------------------

output = {
    "best_model": best_model_name,
    "metrics": results[best_model_name],
    "all_models": results,
    "training_samples": len(X_train),
    "testing_samples": len(X_test)
}

with open(
    os.path.join(MODEL_DIR, "metrics.json"),
    "w"
) as file:
    json.dump(output, file, indent=4)


print("\n--------------------------------")
print("TRAINING COMPLETED SUCCESSFULLY")
print("--------------------------------")

print("Selected model:", best_model_name)
print("Model saved at:", model_path)
print("Metrics saved at: models/metrics.json")
