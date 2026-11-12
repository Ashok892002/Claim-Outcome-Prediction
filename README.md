# Claim Outcome Prediction – End-to-End ML Project

## 1. Project Overview

This project predicts insurance claim outcomes using Machine Learning.

The model classifies insurance claims into three categories:

- Paid
- Partially_Paid
- Denied

A Random Forest classifier is trained using a synthetic dataset and deployed through a FastAPI REST API.

## 2. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest Classifier
- Joblib
- FastAPI
- Uvicorn
- Pydantic

## 3. Dataset Description

The synthetic dataset contains 400 records and 8 columns.

| Column | Description |
|---|---|
| id | Claim identifier |
| payer | Insurance payer |
| provider | Healthcare provider |
| cpt_code | Medical procedure code |
| diagnosis_code | Diagnosis code |
| billed_amount | Claim billed amount |
| patient_age | Patient age |
| status | Claim outcome |

### Target distribution

| Status | Records | Percentage |
|---|---:|---:|
| Paid | 280 | 70% |
| Partially_Paid | 80 | 20% |
| Denied | 40 | 10% |

**Note:** This is synthetic data. The outcome relationships are illustrative assumptions, not actual insurance adjudication rules.

## 4. Feature Engineering

Three additional features were created:

1. `amount_per_year_of_age` – billed amount divided by patient age plus one.
2. `is_high_billed` – identifies claims with billed amounts above 600.
3. `payer_cpt` – combines payer and CPT code.

These features use claim input data and do not use the target status.

## 5. Model Development

- Train/test split: 80% / 20%.
- Stratified sampling to preserve class distribution.
- Categorical encoding using OneHotEncoder.
- Numerical scaling using StandardScaler.
- Random Forest classifier with 200 estimators.
- Class balancing using `class_weight="balanced"`.
- Complete preprocessing and model pipeline saved using Joblib.

## 6. Model Evaluation

Test records: 80

### Classification results

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Paid | 0.75 | 0.64 | 0.69 |
| Partially_Paid | 0.21 | 0.31 | 0.25 |
| Denied | 0.38 | 0.38 | 0.38 |

- Accuracy: 55%
- Macro F1-score: 0.44
- Weighted F1-score: 0.57

### Confusion Matrix

Rows represent actual classes and columns represent predicted classes.

```text
                  Predicted
                Paid  Partial  Denied

Actual Paid       36     17       3
Actual Partial     9      5       2
Actual Denied      3      2       3
```

### Observation

The model performs better on Paid claims.

Partially_Paid is the weakest class, with an F1-score of 0.25. The model frequently misclassifies partially paid claims as paid.

### Possible improvements

- Use a larger and more representative dataset.
- Improve feature engineering.
- Tune Random Forest hyperparameters.
- Evaluate alternative classification algorithms.
- Use cross-validation and class-specific performance analysis.

## 7. FastAPI Deployment

The trained model is deployed using FastAPI.

### Available endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API health check |
| POST | `/predict` | Predict claim outcome |

The prediction endpoint returns:
- Predicted claim status.
- Probability for each outcome class.

Input validation errors are returned with HTTP 400 responses.

## 8. Installation and Execution

### Step 1: Create virtual environment

```bash
python -m venv venv
```

### Step 2: Activate environment (Windows)

```powershell
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Generate dataset

```bash
python generate_data.py
```

### Step 5: Train model

```bash
python train.py
```

### Step 6: Start API server

```bash
uvicorn app:app --reload
```

### Step 7: Open API documentation

Visit:

http://127.0.0.1:8000/docs

Use the POST `/predict` endpoint to test claim predictions.

## 9. Project Structure

```text
Claim-Outcome-Prediction/
│
├── dataset/
│   └── claims_data.csv
│
├── models/
│   └── claim_model.joblib
│
├── generate_data.py
├── features.py
├── train.py
├── app.py
├── requirements.txt
├── README.md
└── venv/
```

## 10. Limitations

- The dataset is synthetic and does not represent real insurance claims.
- Model performance is limited by the size and assumptions of the generated dataset.
- Predicted probabilities are model estimates, not guaranteed claim outcomes.
- The model is intended for demonstration and educational purposes, not real insurance decisions.

## 11. Conclusion

This project demonstrates an end-to-end Machine Learning workflow, including synthetic data generation, feature engineering, model training, evaluation, model persistence and REST API deployment using FastAPI.
