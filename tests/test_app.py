"""
Tests for Healthcare Provider Fraud Detection API
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestInference:
    """Tests for the inference module."""
    
    def test_model_loads(self):
        """Test that the model loads successfully."""
        from src.serving.inference import model
        assert model is not None
    
    def test_predict_returns_string(self):
        """Test that predict returns a string result."""
        from src.serving.inference import predict
        
        # Sample valid input
        sample_data = {
            "count_unique_beneficiary": 100,
            "count_unique_claims": 500,
            "count_dead_beneficiary": 5,
            "count_unique_states": 3,
            "count_unique_counties": 10,
            "mean_hospital_stay_days": 5.0,
            "max_hospital_stay_days": 30,
            "total_top_diagnosis_codes": 10,
            "mean_total_diagnosis": 3.0,
            "mean_total_procedures": 1.5,
            "mean_number_of_physicians": 2.0,
            "mean_difference_stay_vs_claim": 0.0,
            "patients_under_top_attending_physician": 5,
            "patients_under_top_operating_physician": 3,
            "mean_claim_amount": 5000,
            "total_claim_amount": 500000,
            "std_claim_amount": 2000,
            "mean_age": 65,
            "count_of_males": 50,
            "count_of_females": 50,
            "count_of_black_people": 20,
            "count_of_white_people": 60,
            "count_of_hispanic_people": 10,
            "count_of_other_people": 10,
            "count_alzheimer": 10,
            "count_heartfailure": 15,
            "count_kidneydisease": 12,
            "count_cancer": 8,
            "count_obstrpulmonary": 10,
            "count_depression": 20,
            "count_diabetes": 25,
            "count_ischemicheart": 18,
            "count_osteoporasis": 15,
            "count_rheumatoidarthritis": 8,
            "count_stroke": 5,
            "count_renal_disease": 10,
            "mean_annual_inpatient_reimbursement": 10000,
            "mean_annual_inpatient_deductible": 1000,
            "mean_annual_outpatient_reimbursement": 5000,
            "mean_annual_outpatient_deductible": 500,
            "count_inpatient": 50,
            "count_outpatient": 450,
            "claims_per_bene": 5.0
        }
        
        result = predict(sample_data)
        assert isinstance(result, str)
        assert "legitimate" in result.lower() or "fraudulent" in result.lower()
    
    def test_predict_legitimate_provider(self):
        """Test prediction for a typical legitimate provider."""
        from src.serving.inference import predict
        
        # Low-risk provider profile
        low_risk_data = {
            "count_unique_beneficiary": 50,
            "count_unique_claims": 200,
            "count_dead_beneficiary": 2,
            "count_unique_states": 2,
            "count_unique_counties": 5,
            "mean_hospital_stay_days": 4.0,
            "max_hospital_stay_days": 15,
            "total_top_diagnosis_codes": 5,
            "mean_total_diagnosis": 2.0,
            "mean_total_procedures": 1.0,
            "mean_number_of_physicians": 2.0,
            "mean_difference_stay_vs_claim": 0.0,
            "patients_under_top_attending_physician": 2,
            "patients_under_top_operating_physician": 1,
            "mean_claim_amount": 3000,
            "total_claim_amount": 150000,
            "std_claim_amount": 1500,
            "mean_age": 60,
            "count_of_males": 25,
            "count_of_females": 25,
            "count_of_black_people": 10,
            "count_of_white_people": 30,
            "count_of_hispanic_people": 5,
            "count_of_other_people": 5,
            "count_alzheimer": 5,
            "count_heartfailure": 8,
            "count_kidneydisease": 6,
            "count_cancer": 4,
            "count_obstrpulmonary": 5,
            "count_depression": 10,
            "count_diabetes": 12,
            "count_ischemicheart": 8,
            "count_osteoporasis": 7,
            "count_rheumatoidarthritis": 4,
            "count_stroke": 2,
            "count_renal_disease": 5,
            "mean_annual_inpatient_reimbursement": 8000,
            "mean_annual_inpatient_deductible": 800,
            "mean_annual_outpatient_reimbursement": 4000,
            "mean_annual_outpatient_deductible": 400,
            "count_inpatient": 20,
            "count_outpatient": 180,
            "claims_per_bene": 4.0
        }
        
        result = predict(low_risk_data)
        # Just verify it returns a valid result
        assert "legitimate" in result.lower() or "fraudulent" in result.lower()


class TestAPI:
    """Tests for the FastAPI application."""
    
    def test_health_check(self):
        """Test the health check endpoint."""
        from fastapi.testclient import TestClient
        from src.app.main import app
        
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_predict_endpoint(self):
        """Test the predict endpoint with valid data."""
        from fastapi.testclient import TestClient
        from src.app.main import app
        
        client = TestClient(app)
        
        sample_data = {
            "count_unique_beneficiary": 100,
            "count_unique_claims": 500,
            "count_dead_beneficiary": 5,
            "count_unique_states": 3,
            "count_unique_counties": 10,
            "mean_hospital_stay_days": 5.0,
            "max_hospital_stay_days": 30,
            "total_top_diagnosis_codes": 10,
            "mean_total_diagnosis": 3.0,
            "mean_total_procedures": 1.5,
            "mean_number_of_physicians": 2.0,
            "mean_difference_stay_vs_claim": 0.0,
            "patients_under_top_attending_physician": 5,
            "patients_under_top_operating_physician": 3,
            "mean_claim_amount": 5000,
            "total_claim_amount": 500000,
            "std_claim_amount": 2000,
            "mean_age": 65,
            "count_of_males": 50,
            "count_of_females": 50,
            "count_of_black_people": 20,
            "count_of_white_people": 60,
            "count_of_hispanic_people": 10,
            "count_of_other_people": 10,
            "count_alzheimer": 10,
            "count_heartfailure": 15,
            "count_kidneydisease": 12,
            "count_cancer": 8,
            "count_obstrpulmonary": 10,
            "count_depression": 20,
            "count_diabetes": 25,
            "count_ischemicheart": 18,
            "count_osteoporasis": 15,
            "count_rheumatoidarthritis": 8,
            "count_stroke": 5,
            "count_renal_disease": 10,
            "mean_annual_inpatient_reimbursement": 10000,
            "mean_annual_inpatient_deductible": 1000,
            "mean_annual_outpatient_reimbursement": 5000,
            "mean_annual_outpatient_deductible": 500,
            "count_inpatient": 50,
            "count_outpatient": 450,
            "claims_per_bene": 5.0
        }
        
        response = client.post("/predict", json=sample_data)
        
        assert response.status_code == 200
        assert "result" in response.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
