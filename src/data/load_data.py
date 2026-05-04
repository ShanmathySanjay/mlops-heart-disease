import pandas as pd
import os

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

def load_data():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    file_path = os.path.join(base_path, "data", "heart.csv")

    print("Loading from:", file_path)  # DEBUG LINE

    df = pd.read_csv(
        file_path,
        names=COLUMN_NAMES,
        na_values="?"
    )

    print("Dataset loaded successfully\n")
    print("Shape:", df.shape)

    return df

if __name__ == "__main__":
    df = load_data()
    print("\nFirst 5 rows:\n", df.head())