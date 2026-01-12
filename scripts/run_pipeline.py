#!/usr/bin/env python3
"""
Healthcare Provider Fraud Detection - End-to-End ML Pipeline

Usage:
    python scripts/run_pipeline.py
"""

import os
import sys
import joblib
import mlflow 
import mlflow.sklearn 
import pandas as pd
import sklearn.metrics as metrics
import time

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_data, process_beneficiary_data, process_claims_data, process_training_data
from src.utils import validate_inpatient_data, validate_outpatient_data, validate_beneficiary_data
from src.features import generate_top_codes, create_claims_features, merge_beneficiary_and_claims, aggregate_claims_features, merge_provider_train
from src.models import train_model, tune_model, evaluate_model

# === Configuration ===
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

# Create directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def main():
    mlflow.set_tracking_uri("file:///home/nayya/Healthcare_Project/mlruns")
    mlflow.set_experiment("Healthcare Fraud Provider Pipeline")  
    
    with mlflow.start_run(run_name="full_pipeline"):
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("target", "PotentialFraud")
        # =========================================================================
        # STAGE 1: DATA LOADING & VALIDATION
        # =========================================================================
        print("=" * 60)
        print("STAGE 1: DATA LOADING & VALIDATION")
        print("=" * 60)

        t0 = time.time()
        # Load and validate inpatient data
        inpatient = load_data(os.path.join(DATA_DIR, "Train_Inpatientdata-1542865627584.csv"))
        inpatient_val = validate_inpatient_data(inpatient)
        if inpatient_val['success']:
            print(f"✅ Inpatient data validated; {inpatient.shape[0]} rows, {inpatient.shape[1]} columns")
        else:
            raise ValueError("Inpatient data validation failed.")
        mlflow.log_metric("validation_inpatient_pass", int(inpatient_val['success']))

        # Load and validate outpatient data
        outpatient = load_data(os.path.join(DATA_DIR, "Train_Outpatientdata-1542865627584.csv"))
        outpatient_val = validate_outpatient_data(outpatient)
        if outpatient_val['success']:
            print(f"✅ Outpatient data validated; {outpatient.shape[0]} rows, {outpatient.shape[1]} columns")
        else:
            raise ValueError("Outpatient data validation failed.")
        mlflow.log_metric("validation_outpatient_pass", int(outpatient_val['success']))

        # Load and validate beneficiary data
        beneficiary = load_data(os.path.join(DATA_DIR, "Train_Beneficiarydata-1542865627584.csv"))
        beneficiary_val = validate_beneficiary_data(beneficiary)
        if beneficiary_val['success']:
            print(f"✅ Beneficiary data validated; {beneficiary.shape[0]} rows, {beneficiary.shape[1]} columns")
        else:
            raise ValueError("Beneficiary data validation failed.")
        mlflow.log_metric("validation_beneficiary_pass", int(beneficiary_val['success']))

        # Load provider labels
        provider = load_data(os.path.join(DATA_DIR, "Train-1542865627584.csv"))
        print(f"✅ Provider data loaded; {provider.shape[0]} rows, {provider.shape[1]} columns")
        mlflow.log_metric("load_time", time.time() - t0)

        # =========================================================================
        # STAGE 2: DATA PREPROCESSING
        # =========================================================================
        print("\n" + "=" * 60)
        print("STAGE 2: DATA PREPROCESSING")
        print("=" * 60)

        t1 = time.time()

        beneficiary = process_beneficiary_data(beneficiary)
        print("✅ Beneficiary data processed")

        claims = process_claims_data(inpatient, outpatient)
        print(f"✅ Claims data merged; {claims.shape[0]} rows")
        mlflow.log_metric("preprocess_time", time.time() - t1)

        # =========================================================================
        # STAGE 3: FEATURE ENGINEERING
        # =========================================================================
        t2 = time.time()
        print("\n" + "=" * 60)
        print("STAGE 3: FEATURE ENGINEERING")
        print("=" * 60)

        # Generate top codes (returns DataFrames directly)
        top_diagnosis, top_attending, top_operating = generate_top_codes(claims, PROCESSED_DIR)
        print("✅ Top codes generated")

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

        mlflow.log_metric("feature_eng_time", time.time() - t2)
        mlflow.log_metric("n_features", features_train.shape[1] - 2)
        mlflow.log_metric("n_samples", features_train.shape[0])
        mlflow.log_metric("fraud_rate", features_train['PotentialFraud'].mean())

        # Save processed training data
        features_train.to_csv(os.path.join(PROCESSED_DIR, "training_data.csv"), index=False)
        print(f"✅ Training data saved to {PROCESSED_DIR}/training_data.csv")

        # =========================================================================
        # STAGE 4: HYPERPARAMETER TUNING
        # =========================================================================
        t3 = time.time()
        print("\n" + "=" * 60)
        print("STAGE 4: HYPERPARAMETER TUNING")
        print("=" * 60)

        params = tune_model(features_train, 'PotentialFraud')
        print(f"✅ Best params: {params}")
        mlflow.log_metric("tune_time", time.time() - t3)

        # Save best params
        joblib.dump(params, os.path.join(ARTIFACTS_DIR, "best_params.pkl"))
        print(f"✅ Best params saved to {ARTIFACTS_DIR}/best_params.pkl")
        for k, v in params.items():
            mlflow.log_param(f"best_{k}", v)

        # =========================================================================
        # STAGE 5: MODEL TRAINING
        # =========================================================================
        t4 = time.time()
        print("\n" + "=" * 60)
        print("STAGE 5: MODEL TRAINING")
        print("=" * 60)

        model, roc, recall, f1 = train_model(features_train, 'PotentialFraud', params)
        print(f"✅ Model trained with ROC AUC: {roc:.4f}")
        mlflow.log_metric("train_time", time.time() - t4)
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # =========================================================================
        # STAGE 6: SAVE MODEL FOR INFERENCE
        # =========================================================================
        print("\n" + "=" * 60)
        print("STAGE 6: SAVING MODEL FOR INFERENCE")
        print("=" * 60)

        model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact("./artifacts/model.pkl")
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
        print(f"Model Recall: {recall:.4f}")
        print(f"Model F1 Score: {f1:.4f}")
        print(f"Model saved to: {model_path}")
        print(f"To run MLflow UI: mlflow ui")


if __name__ == "__main__":
    main()
