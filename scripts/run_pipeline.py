#!/usr/bin/env python3
"""
Healthcare Provider Fraud Detection - End-to-End ML Pipeline

Usage:
    python scripts/run_pipeline.py
"""

import os
import sys
import joblib

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_data, process_beneficiary_data, process_claims_data, process_training_data
from src.utils import validate_inpatient_data, validate_outpatient_data, validate_beneficiary_data
from src.features import generate_top_codes, create_claims_features, merge_beneficiary_and_claims, aggregate_claims_features, merge_provider_train
from src.models import train_model, tune_model

# === Configuration ===
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

# Create directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def main():
    # =========================================================================
    # STAGE 1: DATA LOADING & VALIDATION
    # =========================================================================
    print("=" * 60)
    print("STAGE 1: DATA LOADING & VALIDATION")
    print("=" * 60)

    # Load and validate inpatient data
    inpatient = load_data(os.path.join(DATA_DIR, "Train_Inpatientdata-1542865627584.csv"))
    inpatient_val = validate_inpatient_data(inpatient)
    if inpatient_val['success']:
        print(f"✅ Inpatient data validated; {inpatient.shape[0]} rows, {inpatient.shape[1]} columns")
    else:
        raise ValueError("Inpatient data validation failed.")

    # Load and validate outpatient data
    outpatient = load_data(os.path.join(DATA_DIR, "Train_Outpatientdata-1542865627584.csv"))
    outpatient_val = validate_outpatient_data(outpatient)
    if outpatient_val['success']:
        print(f"✅ Outpatient data validated; {outpatient.shape[0]} rows, {outpatient.shape[1]} columns")
    else:
        raise ValueError("Outpatient data validation failed.")

    # Load and validate beneficiary data
    beneficiary = load_data(os.path.join(DATA_DIR, "Train_Beneficiarydata-1542865627584.csv"))
    beneficiary_val = validate_beneficiary_data(beneficiary)
    if beneficiary_val['success']:
        print(f"✅ Beneficiary data validated; {beneficiary.shape[0]} rows, {beneficiary.shape[1]} columns")
    else:
        raise ValueError("Beneficiary data validation failed.")

    # Load provider labels
    provider = load_data(os.path.join(DATA_DIR, "Train-1542865627584.csv"))
    print(f"✅ Provider data loaded; {provider.shape[0]} rows, {provider.shape[1]} columns")

    # =========================================================================
    # STAGE 2: DATA PREPROCESSING
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: DATA PREPROCESSING")
    print("=" * 60)

    beneficiary = process_beneficiary_data(beneficiary)
    print("✅ Beneficiary data processed")

    claims = process_claims_data(inpatient, outpatient)
    print(f"✅ Claims data merged; {claims.shape[0]} rows")

    # =========================================================================
    # STAGE 3: FEATURE ENGINEERING
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 3: FEATURE ENGINEERING")
    print("=" * 60)

    # Generate top codes
    generate_top_codes(claims, PROCESSED_DIR)
    print("✅ Top codes generated")

    # Load top codes
    import pandas as pd
    top_diagnosis = pd.read_csv(os.path.join(PROCESSED_DIR, "top_diagnosis_code.csv"))
    top_attending = pd.read_csv(os.path.join(PROCESSED_DIR, "top_attending_physician.csv"))
    top_operating = pd.read_csv(os.path.join(PROCESSED_DIR, "top_operating_physician.csv"))

    # Create claims features
    claims = create_claims_features(claims, top_diagnosis, top_attending, top_operating)
    print("✅ Claims features created")

    # Merge beneficiary and claims
    merged = merge_beneficiary_and_claims(beneficiary, claims)
    print(f"✅ Data merged; {merged.shape[0]} rows")

    # Aggregate to provider level
    features = aggregate_claims_features(merged)
    print(f"✅ Features aggregated; {features.shape[0]} providers, {features.shape[1]} features")

    # Merge with provider labels
    features_train = merge_provider_train(features, provider)
    print(f"✅ Training data ready; {features_train.shape[0]} rows, {features_train.shape[1]} columns")

    # Process training data
    features_train = process_training_data(features_train)

    # Save processed training data
    features_train.to_csv(os.path.join(PROCESSED_DIR, "training_data.csv"), index=False)
    print(f"✅ Training data saved to {PROCESSED_DIR}/training_data.csv")

    # =========================================================================
    # STAGE 4: HYPERPARAMETER TUNING
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 4: HYPERPARAMETER TUNING")
    print("=" * 60)

    params = tune_model(features_train, 'PotentialFraud')
    print(f"✅ Best params: {params}")

    # Save best params
    joblib.dump(params, os.path.join(ARTIFACTS_DIR, "best_params.pkl"))
    print(f"✅ Best params saved to {ARTIFACTS_DIR}/best_params.pkl")

    # =========================================================================
    # STAGE 5: MODEL TRAINING
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 5: MODEL TRAINING")
    print("=" * 60)

    model, roc = train_model(features_train, 'PotentialFraud', params)
    print(f"✅ Model trained with ROC AUC: {roc:.4f}")

    # =========================================================================
    # STAGE 6: SAVE MODEL FOR INFERENCE
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 6: SAVING MODEL FOR INFERENCE")
    print("=" * 60)

    model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

    # Save feature columns for inference
    feature_cols = [col for col in features_train.columns if col not in ['Provider', 'PotentialFraud']]
    joblib.dump(feature_cols, os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))
    print(f"✅ Feature columns saved to {ARTIFACTS_DIR}/feature_columns.pkl")

    # =========================================================================
    # COMPLETE
    # =========================================================================
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Model ROC AUC: {roc:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"To run MLflow UI: mlflow ui")


if __name__ == "__main__":
    main()
