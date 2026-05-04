from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load pipeline (includes preprocessing + model)
model = joblib.load("rf_model.pkl")


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API"}


@app.post("/predict")
def predict(data: dict):
    """
    Input:
    {
        "features": [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]
    }
    """

    features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(features)

    return {"prediction": int(prediction[0])}