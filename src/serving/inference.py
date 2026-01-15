"""
INFERENCE MODULE - Production ML Model Serving
===============================================

This module handles model loading and prediction for the Healthcare Provider
Fraud Detection model.

Supports multiple model formats:
1. Joblib (.pkl) - Simple, fast loading
2. MLflow - Native MLflow format with full metadata

Loading Priority:
1. Docker/Production: /app/model/model.pkl
2. Local exported: src/serving/model/model.pkl  
3. Local artifacts: artifacts/model.pkl
4. MLflow format (fallback)

Usage:
    from serving.inference import predict
    result = predict({"feature1": value, ...})
"""

import os
import joblib
import pandas as pd

# Determine the correct model path based on environment
def _find_model():
    """Find model file in order of priority."""
    
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # Priority 1: Docker production path
    docker_path = "/app/model/model.pkl"
    if os.path.exists(docker_path):
        return docker_path, "joblib"
    
    # Priority 2: Exported model in src/serving/model/
    exported_path = os.path.join(current_dir, "model", "model.pkl")
    if os.path.exists(exported_path):
        return exported_path, "joblib"
    
    # Priority 3: Local artifacts directory
    artifacts_path = os.path.join(project_root, "artifacts", "model.pkl")
    if os.path.exists(artifacts_path):
        return artifacts_path, "joblib"
    
    # Priority 4: MLflow model in exported location
    mlflow_path = os.path.join(current_dir, "model", "mlflow_model")
    if os.path.exists(mlflow_path):
        return mlflow_path, "mlflow"
    
    raise FileNotFoundError(
        "No model found. Please run the training pipeline and export_model.py script.\n"
        f"Searched locations:\n"
        f"  - {docker_path} (Docker)\n"
        f"  - {exported_path} (Exported)\n"
        f"  - {artifacts_path} (Artifacts)\n"
        f"  - {mlflow_path} (MLflow)"
    )


def _load_model():
    """Load the model from the found path."""
    model_path, model_type = _find_model()
    
    if model_type == "joblib":
        model = joblib.load(model_path)
        print(f"✅ Model loaded (joblib) from {model_path}")
    elif model_type == "mlflow":
        import mlflow
        model = mlflow.pyfunc.load_model(model_path)
        print(f"✅ Model loaded (MLflow) from {model_path}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


# Load model at module import time
model = _load_model()


def predict(data: dict) -> str:
    """
    Generate fraud prediction for a healthcare provider.
    
    Args:
        data: Dictionary containing 44 provider features
        
    Returns:
        str: "The healthcare provider is legitimate" or 
             "The healthcare provider is fraudulent"
    
    Raises:
        Exception: If prediction fails
    """
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        result = int(prediction[0])
        
        if result == 0:
            return "The healthcare provider is legitimate"
        else:
            return "The healthcare provider is fraudulent"
    except Exception as e:
        raise Exception(f"Error making prediction: {e}")