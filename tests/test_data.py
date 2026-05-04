from src.data.load_data import load_data

def test_load_data():
    df = load_data()
    
    assert df is not None
    assert not df.empty
    assert "target" in df.columns