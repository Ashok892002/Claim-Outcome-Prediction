
import pandas as pd
import numpy as np
from pathlib import Path

# Reproducibility
np.random.seed(42)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATASET_DIR / "claims_data.csv"

# Dataset size
N = 400

# Generate claim features
payers = ["Aetna", "Cigna", "UnitedHC", "BlueCross", "Medicare"]
providers = ["Dr.Rao", "Dr.Shah", "Dr.Lee", "Dr.Kumar", "Dr.Smith"]
cpt_codes = ["99213", "99214", "99215", "93000", "80053"]
diagnosis_codes = ["M54.5", "J06.9", "E11.9", "I10", "R51"]

df = pd.DataFrame({
    "id": range(1, N + 1),
    "payer": np.random.choice(payers, N),
    "provider": np.random.choice(providers, N),
    "cpt_code": np.random.choice(cpt_codes, N),
    "diagnosis_code": np.random.choice(diagnosis_codes, N),
    "billed_amount": np.round(np.random.uniform(100, 1000, N), 2),
    "patient_age": np.random.randint(18, 86, N)
})

# Synthetic risk assumptions for demonstration only
payer_risk = {
    "Aetna": 0.15,
    "Cigna": 0.25,
    "UnitedHC": 0.45,
    "BlueCross": 0.20,
    "Medicare": 0.35
}

cpt_risk = {
    "99213": 0.10,
    "99214": 0.25,
    "99215": 0.45,
    "93000": 0.30,
    "80053": 0.20
}

diagnosis_risk = {
    "M54.5": 0.30,
    "J06.9": 0.10,
    "E11.9": 0.40,
    "I10": 0.25,
    "R51": 0.15
}

# Calculate a synthetic claim risk score
risk_score = (
    df["payer"].map(payer_risk).to_numpy() * 0.35
    + df["cpt_code"].map(cpt_risk).to_numpy() * 0.30
    + df["diagnosis_code"].map(diagnosis_risk).to_numpy() * 0.15
    + ((df["billed_amount"].to_numpy() - 100) / 900) * 0.20
    + np.random.normal(0, 0.12, N)
)

# Assign exact class counts based on risk ranking
# 280 Paid, 80 Partially_Paid, 40 Denied
ranked_indices = np.argsort(risk_score)[::-1]

status = np.full(N, "Paid", dtype=object)
status[ranked_indices[:40]] = "Denied"
status[ranked_indices[40:120]] = "Partially_Paid"

df["status"] = status

# Save dataset
df.to_csv(OUTPUT_PATH, index=False)

# Display results
print("Dataset generated successfully!")
print(f"Saved to: {OUTPUT_PATH}")
print(f"\nDataset shape: {df.shape}")
print("\nStatus distribution:")
print(df["status"].value_counts())
print("\nStatus percentages:")
print((df["status"].value_counts(normalize=True) * 100).round(2))
print("\nFirst five records:")
print(df.head())
