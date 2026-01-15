#!/usr/bin/env python3
"""
Export Model to src/serving/model/ for Docker builds.

This script copies model artifacts to a committable location.
It supports two sources:
1. MLflow runs (mlruns/*/artifacts/model/) - native MLflow format
2. Local artifacts (artifacts/model.pkl) - joblib format

Usage:
    python scripts/export_model.py

After running, commit src/serving/model/ to Git.
"""

import os
import shutil
import glob

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MLRUNS_PATH = os.path.join(PROJECT_ROOT, "mlruns")
ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, "artifacts")
EXPORT_PATH = os.path.join(PROJECT_ROOT, "src", "serving", "model")


def find_mlflow_model():
    """Find the most recent MLflow model artifact."""
    model_paths = glob.glob(os.path.join(MLRUNS_PATH, "*", "*", "artifacts", "model"))
    
    if not model_paths:
        return None
    
    # Get the most recently modified model
    latest = max(model_paths, key=os.path.getmtime)
    return latest


def export_model():
    """Export model artifacts to src/serving/model/"""
    
    # Clean existing export directory
    if os.path.exists(EXPORT_PATH):
        print(f"Removing existing {EXPORT_PATH}")
        shutil.rmtree(EXPORT_PATH)
    
    os.makedirs(EXPORT_PATH, exist_ok=True)
    
    # Try MLflow model first
    mlflow_model = find_mlflow_model()
    if mlflow_model:
        print(f"Found MLflow model: {mlflow_model}")
        # Copy entire MLflow model directory
        shutil.copytree(mlflow_model, os.path.join(EXPORT_PATH, "mlflow_model"))
        print("✅ Copied MLflow model")
        
        # Also get feature_columns.txt from the same run
        run_artifacts = os.path.dirname(mlflow_model)
        feature_txt = os.path.join(run_artifacts, "feature_columns.txt")
        if os.path.exists(feature_txt):
            shutil.copy(feature_txt, os.path.join(EXPORT_PATH, "feature_columns.txt"))
            print("✅ Copied feature_columns.txt from MLflow")
    
    # Also copy joblib model from artifacts/ (for simpler loading)
    joblib_model = os.path.join(ARTIFACTS_PATH, "model.pkl")
    if os.path.exists(joblib_model):
        shutil.copy(joblib_model, os.path.join(EXPORT_PATH, "model.pkl"))
        print("✅ Copied model.pkl from artifacts/")
    
    # Copy feature columns pickle
    features_pkl = os.path.join(ARTIFACTS_PATH, "feature_columns.pkl")
    if os.path.exists(features_pkl):
        shutil.copy(features_pkl, os.path.join(EXPORT_PATH, "feature_columns.pkl"))
        print("✅ Copied feature_columns.pkl")
    
    # Copy best params
    params_pkl = os.path.join(ARTIFACTS_PATH, "best_params.pkl")
    if os.path.exists(params_pkl):
        shutil.copy(params_pkl, os.path.join(EXPORT_PATH, "best_params.pkl"))
        print("✅ Copied best_params.pkl")
    
    # Verify we have at least one model
    has_mlflow = os.path.exists(os.path.join(EXPORT_PATH, "mlflow_model"))
    has_joblib = os.path.exists(os.path.join(EXPORT_PATH, "model.pkl"))
    
    if not has_mlflow and not has_joblib:
        raise FileNotFoundError(
            "No model found to export. Run the training pipeline first:\n"
            "  python scripts/run_pipeline.py"
        )
    
    # Summary
    print(f"\n✅ Model exported to {EXPORT_PATH}")
    print("\nExported files:")
    for root, dirs, files in os.walk(EXPORT_PATH):
        level = root.replace(EXPORT_PATH, '').count(os.sep)
        indent = '  ' * level
        folder = os.path.basename(root)
        if level > 0:
            print(f"{indent}{folder}/")
        for file in files:
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath)
            print(f"{indent}  - {file} ({size:,} bytes)")
    
    print("\n📝 Next steps:")
    print("  1. git add src/serving/model/")
    print("  2. git commit -m 'Export model for Docker'")
    print("  3. git push")


if __name__ == "__main__":
    export_model()
