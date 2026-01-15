"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
=========================================================================

This application provides a complete serving solution for the Healthcare Provider
Fraud Detection model with both programmatic API access and a user-friendly web interface.

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation
- Gradio: User-friendly web UI for manual testing and demonstrations
- Pydantic: Data validation and automatic API documentation

Endpoints:
- GET  /        : Health check (for load balancers)
- POST /predict : JSON API for predictions
- GET  /ui      : Gradio web interface

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys
import gradio as gr

# Ensure imports work from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.inference import predict


# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

app = FastAPI(
    title="Healthcare Provider Fraud Detection API",
    description="ML API for predicting healthcare provider fraud using XGBoost",
    version="1.0.0"
)


# =============================================================================
# PYDANTIC SCHEMA - Request Validation
# =============================================================================

class ProviderFeatures(BaseModel):
    """
    Provider features schema for fraud prediction.
    
    This schema defines the 44 aggregated features required for fraud prediction.
    Features are computed at the provider level from claims, beneficiary, and 
    billing data.
    
    Feature Categories:
    - Claims statistics (counts, unique values)
    - Hospital stay metrics (mean, max days)
    - Billing amounts (claim amounts, reimbursements)
    - Demographics (age, gender, race distributions)
    - Chronic conditions (disease counts)
    """
    # Claims Statistics
    count_unique_beneficiary: float      # Number of unique patients
    count_unique_claims: float           # Total number of claims
    count_dead_beneficiary: float        # Deceased patients count
    count_unique_states: float           # Geographic spread (states)
    count_unique_counties: float         # Geographic spread (counties)
    claims_per_bene: float               # Claims per beneficiary ratio
    count_inpatient: float               # Inpatient claims count
    count_outpatient: float              # Outpatient claims count
    
    # Hospital Stay Metrics
    mean_hospital_stay_days: float       # Average hospital stay duration
    max_hospital_stay_days: float        # Maximum hospital stay duration
    mean_difference_stay_vs_claim: float # Billing vs stay discrepancy
    
    # Diagnosis & Procedures
    total_top_diagnosis_codes: float     # Common diagnosis codes count
    mean_total_diagnosis: float          # Average diagnoses per claim
    mean_total_procedures: float         # Average procedures per claim
    mean_number_of_physicians: float     # Average physicians per claim
    patients_under_top_attending_physician: float   # Top attending physician patients
    patients_under_top_operating_physician: float   # Top operating physician patients
    
    # Billing & Financial
    mean_claim_amount: float             # Average claim reimbursement
    total_claim_amount: float            # Total claim reimbursement
    std_claim_amount: float              # Claim amount variability
    mean_annual_inpatient_reimbursement: float   # Annual IP reimbursement
    mean_annual_inpatient_deductible: float      # Annual IP deductible
    mean_annual_outpatient_reimbursement: float  # Annual OP reimbursement
    mean_annual_outpatient_deductible: float     # Annual OP deductible
    
    # Demographics
    mean_age: float                      # Average patient age
    count_of_males: float                # Male patient count
    count_of_females: float              # Female patient count
    count_of_black_people: float         # Black patient count
    count_of_white_people: float         # White patient count
    count_of_hispanic_people: float      # Hispanic patient count
    count_of_other_people: float         # Other race patient count
    
    # Chronic Conditions
    count_alzheimer: float               # Alzheimer's disease count
    count_heartfailure: float            # Heart failure count
    count_kidneydisease: float           # Kidney disease count
    count_cancer: float                  # Cancer count
    count_obstrpulmonary: float          # COPD count
    count_depression: float              # Depression count
    count_diabetes: float                # Diabetes count
    count_ischemicheart: float           # Ischemic heart disease count
    count_osteoporasis: float            # Osteoporosis count
    count_rheumatoidarthritis: float     # Rheumatoid arthritis count
    count_stroke: float                  # Stroke count
    count_renal_disease: float           # Renal disease count


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
def health_check():
    """
    Health check endpoint for monitoring and load balancer health checks.
    
    Returns:
        dict: Status indicator for service health
    """
    return {"status": "ok"}


@app.post("/predict")
def get_prediction(features: ProviderFeatures):
    """
    Main prediction endpoint for healthcare provider fraud detection.
    
    This endpoint:
    1. Receives validated provider data via Pydantic model
    2. Calls the inference pipeline to generate prediction
    3. Returns fraud prediction in JSON format
    
    Args:
        features: ProviderFeatures object with 44 aggregated features
        
    Returns:
        dict: {"result": "The healthcare provider is legitimate/fraudulent"}
        dict: {"error": "error_message"} if prediction fails
    """
    try:
        features_dict = features.model_dump()
        output = predict(features_dict)
        return {"result": output}
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# GRADIO WEB INTERFACE
# =============================================================================

def gradio_interface(
    count_unique_beneficiary, count_unique_claims, count_dead_beneficiary, 
    count_unique_states, count_unique_counties, mean_hospital_stay_days, 
    max_hospital_stay_days, total_top_diagnosis_codes, mean_total_diagnosis, 
    mean_total_procedures, mean_number_of_physicians, mean_difference_stay_vs_claim, 
    patients_under_top_attending_physician, patients_under_top_operating_physician, 
    mean_claim_amount, total_claim_amount, std_claim_amount, mean_age, 
    count_of_males, count_of_females, count_of_black_people, count_of_white_people, 
    count_of_hispanic_people, count_of_other_people, count_alzheimer, count_heartfailure, 
    count_kidneydisease, count_cancer, count_obstrpulmonary, count_depression, 
    count_diabetes, count_ischemicheart, count_osteoporasis, count_rheumatoidarthritis, 
    count_stroke, count_renal_disease, mean_annual_inpatient_reimbursement, 
    mean_annual_inpatient_deductible, mean_annual_outpatient_reimbursement, 
    mean_annual_outpatient_deductible, count_inpatient, count_outpatient, claims_per_bene
):
    """
    Gradio interface function that processes form inputs and returns prediction.
    
    This function:
    1. Takes individual form inputs from Gradio UI
    2. Constructs the data dictionary matching the API schema
    3. Calls the same inference pipeline used by the API
    4. Returns user-friendly prediction string
    """
    payload = {
        "count_unique_beneficiary": float(count_unique_beneficiary),
        "count_unique_claims": float(count_unique_claims),
        "count_dead_beneficiary": float(count_dead_beneficiary),
        "count_unique_states": float(count_unique_states),
        "count_unique_counties": float(count_unique_counties),
        "mean_hospital_stay_days": float(mean_hospital_stay_days),
        "max_hospital_stay_days": float(max_hospital_stay_days),
        "total_top_diagnosis_codes": float(total_top_diagnosis_codes),
        "mean_total_diagnosis": float(mean_total_diagnosis),
        "mean_total_procedures": float(mean_total_procedures),
        "mean_number_of_physicians": float(mean_number_of_physicians),
        "mean_difference_stay_vs_claim": float(mean_difference_stay_vs_claim),
        "patients_under_top_attending_physician": float(patients_under_top_attending_physician),
        "patients_under_top_operating_physician": float(patients_under_top_operating_physician),
        "mean_claim_amount": float(mean_claim_amount),
        "total_claim_amount": float(total_claim_amount),
        "std_claim_amount": float(std_claim_amount),
        "mean_age": float(mean_age),
        "count_of_males": float(count_of_males),
        "count_of_females": float(count_of_females),
        "count_of_black_people": float(count_of_black_people),
        "count_of_white_people": float(count_of_white_people),
        "count_of_hispanic_people": float(count_of_hispanic_people),
        "count_of_other_people": float(count_of_other_people),
        "count_alzheimer": float(count_alzheimer),
        "count_heartfailure": float(count_heartfailure),
        "count_kidneydisease": float(count_kidneydisease),
        "count_cancer": float(count_cancer),
        "count_obstrpulmonary": float(count_obstrpulmonary),
        "count_depression": float(count_depression),
        "count_diabetes": float(count_diabetes),
        "count_ischemicheart": float(count_ischemicheart),
        "count_osteoporasis": float(count_osteoporasis),
        "count_rheumatoidarthritis": float(count_rheumatoidarthritis),
        "count_stroke": float(count_stroke),
        "count_renal_disease": float(count_renal_disease),
        "mean_annual_inpatient_reimbursement": float(mean_annual_inpatient_reimbursement),
        "mean_annual_inpatient_deductible": float(mean_annual_inpatient_deductible),
        "mean_annual_outpatient_reimbursement": float(mean_annual_outpatient_reimbursement),
        "mean_annual_outpatient_deductible": float(mean_annual_outpatient_deductible),
        "count_inpatient": float(count_inpatient),
        "count_outpatient": float(count_outpatient),
        "claims_per_bene": float(claims_per_bene)
    }
    
    result = predict(payload)
    return str(result)


# Build Gradio interface with organized inputs
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        # Claims Statistics
        gr.Number(label="Unique Beneficiaries", value=100),
        gr.Number(label="Unique Claims", value=500),
        gr.Number(label="Dead Beneficiaries", value=5),
        gr.Number(label="Unique States", value=3),
        gr.Number(label="Unique Counties", value=10),
        gr.Number(label="Mean Hospital Stay (days)", value=5.0),
        gr.Number(label="Max Hospital Stay (days)", value=30),
        gr.Number(label="Top Diagnosis Codes", value=10),
        gr.Number(label="Mean Diagnoses", value=3.0),
        gr.Number(label="Mean Procedures", value=1.5),
        gr.Number(label="Mean Physicians", value=2.0),
        gr.Number(label="Stay vs Claim Difference", value=0.0),
        gr.Number(label="Top Attending Physician Patients", value=5),
        gr.Number(label="Top Operating Physician Patients", value=3),
        gr.Number(label="Mean Claim Amount ($)", value=5000),
        gr.Number(label="Total Claim Amount ($)", value=500000),
        gr.Number(label="Std Claim Amount ($)", value=2000),
        gr.Number(label="Mean Age", value=65),
        gr.Number(label="Males", value=50),
        gr.Number(label="Females", value=50),
        gr.Number(label="Black Patients", value=20),
        gr.Number(label="White Patients", value=60),
        gr.Number(label="Hispanic Patients", value=10),
        gr.Number(label="Other Race Patients", value=10),
        gr.Number(label="Alzheimer Cases", value=10),
        gr.Number(label="Heart Failure Cases", value=15),
        gr.Number(label="Kidney Disease Cases", value=12),
        gr.Number(label="Cancer Cases", value=8),
        gr.Number(label="COPD Cases", value=10),
        gr.Number(label="Depression Cases", value=20),
        gr.Number(label="Diabetes Cases", value=25),
        gr.Number(label="Ischemic Heart Cases", value=18),
        gr.Number(label="Osteoporosis Cases", value=15),
        gr.Number(label="Rheumatoid Arthritis Cases", value=8),
        gr.Number(label="Stroke Cases", value=5),
        gr.Number(label="Renal Disease Cases", value=10),
        gr.Number(label="Mean IP Reimbursement ($)", value=10000),
        gr.Number(label="Mean IP Deductible ($)", value=1000),
        gr.Number(label="Mean OP Reimbursement ($)", value=5000),
        gr.Number(label="Mean OP Deductible ($)", value=500),
        gr.Number(label="Inpatient Count", value=50),
        gr.Number(label="Outpatient Count", value=450),
        gr.Number(label="Claims per Beneficiary", value=5.0)
    ],
    outputs=gr.Textbox(label="Fraud Prediction", lines=2),
    title="🏥 Healthcare Provider Fraud Detection",
    description="""
    **Predict healthcare provider fraud using machine learning**
    
    Enter the aggregated provider statistics below to get a fraud prediction. 
    The model uses XGBoost trained on Medicare claims data to identify 
    providers at risk of fraudulent billing practices.
    
    💡 **Tip**: Providers with unusually high claim amounts, many deceased patients,
    or billing anomalies tend to have higher fraud risk.
    """,
    examples=[
        # High fraud risk example
        [200, 1500, 25, 10, 50, 8.0, 60, 30, 5.0, 3.0, 4.0, 5.0, 20, 15,
         12000, 1800000, 8000, 72, 80, 120, 40, 100, 30, 30, 25, 35, 30, 
         20, 25, 40, 50, 40, 30, 20, 15, 25, 25000, 2500, 12000, 1200,
         150, 1350, 7.5],
        # Low fraud risk example
        [50, 200, 2, 2, 5, 4.0, 15, 5, 2.0, 1.0, 2.0, 0.0, 2, 1,
         3000, 150000, 1500, 60, 25, 25, 10, 30, 5, 5, 5, 8, 6, 
         4, 5, 10, 12, 8, 7, 4, 2, 5, 8000, 800, 4000, 400,
         20, 180, 4.0]
    ],
    theme=gr.themes.Soft()
)


# =============================================================================
# MOUNT GRADIO ON FASTAPI
# =============================================================================

# IMPORTANT: This must be the final line to properly integrate Gradio with FastAPI
app = gr.mount_gradio_app(app, demo, path="/ui")
