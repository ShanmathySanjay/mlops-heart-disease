import joblib
import pandas as pd

def test_model_prediction():
    model = joblib.load("rf_model.pkl")

    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal"
    ]

    sample = pd.DataFrame([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]], columns=columns)

    pred = model.predict(sample)

    assert pred[0] in [0, 1]