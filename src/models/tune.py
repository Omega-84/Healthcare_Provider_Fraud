import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier
import sklearn.metrics as metrics
import optuna

def tune_model(df: pd.DataFrame, target: str):
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

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "auc"
        }
        
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        pipeline = make_pipeline(preprocessor, model)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
        
        return cv_scores.mean() 
    
    study = optuna.create_study(
    direction='maximize',  # Maximize AUC
    study_name='xgboost_fraud_detection_prod')

    study.optimize(objective, n_trials=100, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({
    "random_state": 42,
    "n_jobs": -1,
    "scale_pos_weight": scale_pos_weight,
    "eval_metric": "logloss"
    })

    print("Best Params:", best_params)
    return best_params