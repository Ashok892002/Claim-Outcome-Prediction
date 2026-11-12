
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "claim_model.joblib"

# Load trained pipeline
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Claim Outcome Prediction API",
    description="Predicts insurance claim outcomes using a trained ML model.",
    version="1.0"
)


# Convert validation errors to HTTP 400
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors()}
    )


# Request schema
class ClaimInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    payer: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    cpt_code: str = Field(min_length=1)
    diagnosis_code: str = Field(min_length=1)
    billed_amount: float = Field(gt=0)
    patient_age: int = Field(ge=1, le=120)


@app.get("/")
def home():
    return {
        "message": "Claim Outcome Prediction API is running"
    }


@app.post("/predict")
def predict_claim(claim: ClaimInput):
    try:
        input_data = pd.DataFrame([claim.model_dump()])

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        probability_result = {
            str(label): round(float(probability), 4)
            for label, probability in zip(
                model.classes_,
                probabilities
            )
        }

        return {
            "predicted_status": str(prediction),
            "probabilities": probability_result
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
