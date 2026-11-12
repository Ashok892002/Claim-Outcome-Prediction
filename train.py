from features import add_features
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "claims_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "claim_model.joblib"


# Feature engineering
def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded:", df.shape)

    # Separate features and target
    X = df.drop(columns=["id", "status"])
    y = df["status"]

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Feature columns
    categorical_features = [
        "payer",
        "provider",
        "cpt_code",
        "diagnosis_code",
        "payer_cpt"
    ]

    numerical_features = [
        "billed_amount",
        "patient_age",
        "amount_per_year_of_age",
        "is_high_billed"
    ]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_features
            )
        ]
    )

    # Complete ML pipeline
    model = Pipeline(
        steps=[
            (
                "feature_engineering",
                FunctionTransformer(
                    add_features,
                    validate=False
                )
            ),
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                )
            )
        ]
    )

    # Train model
    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=["Paid", "Partially_Paid", "Denied"],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=["Paid", "Partially_Paid", "Denied"]
        )
    )

    # Save complete pipeline
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully!")
    print("Model path:", MODEL_PATH)

    print("\nTrain records:", len(X_train))
    print("Test records:", len(X_test))


if __name__ == "__main__":
    main()
