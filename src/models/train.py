import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def train_model(df: pd.DataFrame, target: str, xgb_params: dict = None):
    """
    Train XGBoost model.
    
    Args:
        df: Training DataFrame
        target: Target column name
        xgb_params: Optional XGBoost parameters (from tuning). 
                    If None, uses defaults.
    
    Returns:
        tuple: (trained_pipeline, roc_auc_score)
    """

    X = df.drop(['Provider',target],axis=1)
    y = df[target]

    standard_scaling_column = ['mean_age','mean_claim_amount','total_claim_amount','std_claim_amount','mean_hospital_stay_days',
                           'max_hospital_stay_days','mean_annual_inpatient_reimbursement','mean_annual_inpatient_deductible',
                           'mean_annual_outpatient_reimbursement',  'mean_annual_outpatient_deductible']

    min_max_cols = [
    'count_unique_beneficiary', 'count_unique_claims', 'claims_per_bene',
    'count_unique_states', 'count_unique_counties',
    'total_top_diagnosis_codes', 'mean_total_diagnosis', 'mean_total_procedures',
    'mean_number_of_physicians', 'mean_difference_stay_vs_claim',
    'count_inpatient', 'count_outpatient',
    'count_of_males', 'count_of_females', 'count_of_black_people', 'count_of_white_people' ,
    'count_of_hispanic_people', 'count_of_other_people'
        ]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

    preprocessor = make_column_transformer(
    (StandardScaler(), standard_scaling_column),
    (MinMaxScaler(), min_max_cols),
    remainder='passthrough')

    scale_pos_weight = round((y_train==0).sum()/(y_train==1).sum())

    # Default XGBoost parameters
    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_jobs': -1,
        'random_state': 42,
        'scale_pos_weight': scale_pos_weight
    }
    
    # Merge with custom params if provided
    if xgb_params is not None:
        default_params.update(xgb_params)
        default_params['scale_pos_weight'] = scale_pos_weight  # Always recalculate
    
    xgb = XGBClassifier(**default_params)

    pipe_xgb = make_pipeline(preprocessor, xgb)

    # Train model
    pipe_xgb.fit(X_train, y_train)
    preds_proba = pipe_xgb.predict_proba(X_test)[:, 1]  # For ROC-AUC
    preds = pipe_xgb.predict(X_test)  # Binary predictions for recall/f1
    roc = roc_auc_score(y_test, preds_proba)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    return pipe_xgb, roc, recall, f1

