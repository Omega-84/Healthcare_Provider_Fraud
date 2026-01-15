from fastapi import FastAPI
from pydantic import BaseModel
import os
import gradio as gr
import sys

# Ensure imports work from src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.inference import predict

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
def get_prediction(features: Features):
    features_dict = features.model_dump()
    try:
        output = predict(features_dict)
        return {"result": output}
    except Exception as e:
        return {"error": str(e)}


def gradio_interface(
    count_unique_beneficiary, count_unique_claims, count_dead_beneficiary, count_unique_states, count_unique_counties, 
    mean_hospital_stay_days, max_hospital_stay_days, total_top_diagnosis_codes, mean_total_diagnosis, mean_total_procedures, mean_number_of_physicians,
    mean_difference_stay_vs_claim, patients_under_top_attending_physician, patients_under_top_operating_physician, mean_claim_amount, total_claim_amount, 
    std_claim_amount, mean_age, count_of_males, count_of_females, count_of_black_people, count_of_white_people, count_of_hispanic_people, count_of_other_people, 
    count_alzheimer, count_heartfailure, count_kidneydisease, count_cancer, count_obstrpulmonary, count_depression, count_diabetes, count_ischemicheart, 
    count_osteoporasis, count_rheumatoidarthritis, count_stroke, count_renal_disease, mean_annual_inpatient_reimbursement, mean_annual_inpatient_deductible,
mean_annual_outpatient_reimbursement, mean_annual_outpatient_deductible, count_inpatient, count_outpatient, claims_per_bene
):
    payload = {
        "count_unique_beneficiary": count_unique_beneficiary,
        "count_unique_claims": count_unique_claims,
        "count_dead_beneficiary": count_dead_beneficiary,
        "count_unique_states": count_unique_states,
        "count_unique_counties": count_unique_counties,
        "mean_hospital_stay_days": mean_hospital_stay_days,
        "max_hospital_stay_days": max_hospital_stay_days,
        "total_top_diagnosis_codes": total_top_diagnosis_codes,
        "mean_total_diagnosis": mean_total_diagnosis,
        "mean_total_procedures": mean_total_procedures,
        "mean_number_of_physicians": mean_number_of_physicians,
        "mean_difference_stay_vs_claim": mean_difference_stay_vs_claim,
        "patients_under_top_attending_physician": patients_under_top_attending_physician,
        "patients_under_top_operating_physician": patients_under_top_operating_physician,
        "mean_claim_amount": mean_claim_amount,
        "total_claim_amount": total_claim_amount,
        "std_claim_amount": std_claim_amount,
        "mean_age": mean_age,
        "count_of_males": count_of_males,
        "count_of_females": count_of_females,
        "count_of_black_people": count_of_black_people,
        "count_of_white_people": count_of_white_people,
        "count_of_hispanic_people": count_of_hispanic_people,
        "count_of_other_people": count_of_other_people,
        "count_alzheimer": count_alzheimer,
        "count_heartfailure": count_heartfailure,
        "count_kidneydisease": count_kidneydisease,
        "count_cancer": count_cancer,
        "count_obstrpulmonary": count_obstrpulmonary,
        "count_depression": count_depression,
        "count_diabetes": count_diabetes,
        "count_ischemicheart": count_ischemicheart,
        "count_osteoporasis": count_osteoporasis,
        "count_rheumatoidarthritis": count_rheumatoidarthritis,
        "count_stroke": count_stroke,
        "count_renal_disease": count_renal_disease,
        "mean_annual_inpatient_reimbursement": mean_annual_inpatient_reimbursement,
        "mean_annual_inpatient_deductible": mean_annual_inpatient_deductible,
        "mean_annual_outpatient_reimbursement": mean_annual_outpatient_reimbursement,
        "mean_annual_outpatient_deductible": mean_annual_outpatient_deductible,
        "count_inpatient": count_inpatient,
        "count_outpatient": count_outpatient,
        "claims_per_bene": claims_per_bene
    }
    out = predict(payload)
    return out

demo = gr.Interface(
    fn = gradio_interface,
    inputs=[
        gr.Number(label = "Count of unique beneficiaries"),
        gr.Number(label = "Count of unique claims"),
        gr.Number(label = "Count of dead beneficiaries"),
        gr.Number(label = "Count of unique states"),
        gr.Number(label = "Count of unique counties"),
        gr.Number(label = "Mean hospital stay days"),
        gr.Number(label = "Max hospital stay days"),
        gr.Number(label = "Total top diagnosis codes"),
        gr.Number(label = "Mean total diagnosis"),
        gr.Number(label = "Mean total procedures"),
        gr.Number(label = "Mean number of physicians"),
        gr.Number(label = "Mean difference stay vs claim"),
        gr.Number(label = "Patients under top attending physician"),
        gr.Number(label = "Patients under top operating physician"),
        gr.Number(label = "Mean claim amount"),
        gr.Number(label = "Total claim amount"),
        gr.Number(label = "Std claim amount"),
        gr.Number(label = "Mean age"),
        gr.Number(label = "Count of males"),
        gr.Number(label = "Count of females"),
        gr.Number(label = "Count of black people"),
        gr.Number(label = "Count of white people"),
        gr.Number(label = "Count of hispanic people"),
        gr.Number(label = "Count of other people"),
        gr.Number(label = "Count of alzheimer"),
        gr.Number(label = "Count of heartfailure"),
        gr.Number(label = "Count of kidneydisease"),
        gr.Number(label = "Count of cancer"),
        gr.Number(label = "Count of obstrpulmonary"),
        gr.Number(label = "Count of depression"),
        gr.Number(label = "Count of diabetes"),
        gr.Number(label = "Count of ischemicheart"),
        gr.Number(label = "Count of osteoporasis"),
        gr.Number(label = "Count of rheumatoidarthritis"),
        gr.Number(label = "Count of stroke"),
        gr.Number(label = "Count of renal disease"),
        gr.Number(label = "Mean annual inpatient reimbursement"),
        gr.Number(label = "Mean annual inpatient deductible"),
        gr.Number(label = "Mean annual outpatient reimbursement"),
        gr.Number(label = "Mean annual outpatient deductible"),
        gr.Number(label = "Count of inpatient"),
        gr.Number(label = "Count of outpatient"),
        gr.Number(label = "Claims per bene")
    ],
    outputs="text",
    title="Healthcare Provider Fraud Prediction",
    description="Predict if a healthcare provider is fraudulent or legitimate based on the given features"
    )

app = gr.mount_gradio_app(app, demo, path="/ui")
    
