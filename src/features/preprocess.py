import pandas as pd

def preprocess_data(df):
    print("Before cleaning:", df.shape)

    # 1. Handle missing values
    df = df.dropna()

    print("After dropping missing:", df.shape)

    # 2. Convert target to binary
    df.loc[:, "target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    print("\nTarget value counts:\n", df["target"].value_counts())

    # 3. Split features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y


if __name__ == "__main__":
    from src.data.load_data import load_data

    df = load_data()
    X, y = preprocess_data(df)

    print("\nFinal Features shape:", X.shape)
    print("Final Target shape:", y.shape)