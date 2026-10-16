import json
import os

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from predict import predict_data

METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "metrics.json")

app = FastAPI(
    title="Diabetes Progression API",
    description="Predicts a diabetes progression score using a Linear Regression model.",
)


class DiabetesData(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 0.0381,
                    "sex": 0.0507,
                    "bmi": 0.0617,
                    "bp": 0.0219,
                    "s1": -0.0442,
                    "s2": -0.0348,
                    "s3": -0.0434,
                    "s4": -0.0026,
                    "s5": 0.0199,
                    "s6": -0.0176,
                }
            ]
        }
    }


class DiabetesResponse(BaseModel):
    predicted_progression: float


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    """Check that the API is running."""
    return {"status": "healthy"}


@app.get("/model_info")
async def model_info():
    """Return the model type, dataset and test set metrics."""
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return {"model": "LinearRegression", "dataset": "scikit-learn diabetes", "test_metrics": metrics}


@app.post("/predict", response_model=DiabetesResponse)
async def predict_progression(data: DiabetesData):
    """Predict the diabetes progression score for one patient."""
    try:
        features = [[data.age, data.sex, data.bmi, data.bp, data.s1,
                     data.s2, data.s3, data.s4, data.s5, data.s6]]
        prediction = predict_data(features)
        return DiabetesResponse(predicted_progression=round(float(prediction[0]), 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
