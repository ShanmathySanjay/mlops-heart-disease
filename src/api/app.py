import logging
import pandas as pd
from fastapi import FastAPI
import joblib
from prometheus_client import Counter, generate_latest
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter("request_count", "Total prediction requests")

app = FastAPI()

model = joblib.load("rf_model.pkl")


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API"}


@app.post("/predict")
def predict(data: dict):
    try:
        features = data["features"]

        columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal"
        ]

        df = pd.DataFrame([features], columns=columns)

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        logger.info(f"Input: {features}")
        logger.info(f"Prediction: {prediction}, Confidence: {probability}")

        REQUEST_COUNT.inc()

        return {
            "prediction": int(prediction),
            "confidence": float(probability)
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")