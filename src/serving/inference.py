import pandas as pd 
import joblib
import os

# Model path - use environment variable or default to local artifacts
MODEL_PATH = os.environ.get("MODEL_PATH", "./artifacts/model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model from {MODEL_PATH}: {e}")
    raise Exception(f"Failed to load model: {e}")

def predict(data: dict) -> str:
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df).tolist()
        result = prediction[0]
        if result == 0:
            return "The healthcare provider is legitimate"
        else:
            return "The healthcare provider is fraudulent"
    except Exception as e:
        raise Exception(f"Error making prediction: {e}")