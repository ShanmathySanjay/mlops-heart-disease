from src.data.load_data import load_data
from src.features.preprocess import preprocess_data

def test_preprocess():
    df = load_data()
    X, y = preprocess_data(df)

    assert X is not None
    assert y is not None
    assert len(X) == len(y)