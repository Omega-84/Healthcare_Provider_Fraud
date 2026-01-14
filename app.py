from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

model = joblib.load('artifacts/model.pkl')

class Features(BaseModel):
    count_unique_beneficiary: float
    count_unique_claims: float
    count_dead_beneficiary: float
    count_unique_states: float
    count_unique_counties: float
    mean_hospital_stay_days: float
    max_hospital_stay_days: float
    total_top_diagnosis_codes: float
    mean_total_diagnosis: float
    mean_total_procedures: float
    mean_number_of_physicians: float
    mean_difference_stay_vs_claim: float
    patients_under_top_attending_physician: float
    patients_under_top_operating_physician: float
    mean_claim_amount: float
    total_claim_amount: float
    std_claim_amount: float
    mean_age: float
    count_of_males: float
    count_of_females: float
    count_of_black_people: float
    count_of_white_people: float
    count_of_hispanic_people: float
    count_of_other_people: float
    count_alzheimer: float
    count_heartfailure: float
    count_kidneydisease: float
    count_cancer: float
    count_obstrpulmonary: float
    count_depression: float
    count_diabetes: float
    count_ischemicheart: float
    count_osteoporasis: float
    count_rheumatoidarthritis: float
    count_stroke: float
    count_renal_disease: float
    mean_annual_inpatient_reimbursement: float
    mean_annual_inpatient_deductible: float
    mean_annual_outpatient_reimbursement: float
    mean_annual_outpatient_deductible: float
    count_inpatient: float
    count_outpatient: float
    claims_per_bene: float

app = FastAPI(title='Healthcare Provider Fraud Prediction')

@app.get("/")
def home():
    return {"message": "Welcome to the Healthcare Provider Fraud Prediction API"}

@app.post("/predict")
def predict(features: Features):
    features = features.model_dump()
    features_df = pd.DataFrame([features])
    prediction = int(model.predict(features_df)[0])
    if prediction == 0:
        out = "The healthcare provider is legitimate"
    else:
        out = "The healthcare provider is fraudulent"

    return {"result": prediction,"prediction": out}
