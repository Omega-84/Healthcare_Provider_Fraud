"""
Data Validation Module.

This module provides validation functions for healthcare datasets
to ensure data quality before processing and training.
"""

import pandas as pd
from typing import Dict, List


# =============================================================================
# INPATIENT DATA VALIDATION
# =============================================================================

def validate_inpatient_data(df: pd.DataFrame, raise_on_fail: bool = False) -> Dict:
    """
    Validate inpatient claims data.
    
    Args:
        df: Inpatient claims DataFrame
        
    Returns:
        dict: Validation report with success status and details
    """
    results = []
    
    # Required columns
    required_columns = [
        'BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
       'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'AdmissionDt', 'ClmAdmitDiagnosisCode',
       'DeductibleAmtPaid', 'DischargeDt', 'DiagnosisGroupCode',
       'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3',
       'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6',
       'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
       'ClmDiagnosisCode_10', 'ClmProcedureCode_1', 'ClmProcedureCode_2',
       'ClmProcedureCode_3', 'ClmProcedureCode_4', 'ClmProcedureCode_5',
       'ClmProcedureCode_6'
    ]
    
    for col in required_columns:
        success = col in df.columns
        results.append({
            'test': 'column_exists',
            'column': col,
            'success': success,
            'message': f"Column '{col}' exists" if success else f"Missing column: {col}"
        })
    
    # ClaimID uniqueness
    if 'ClaimID' in df.columns:
        is_unique = df['ClaimID'].nunique() == len(df)
        results.append({
            'test': 'unique_values',
            'column': 'ClaimID',
            'success': is_unique,
            'message': f"ClaimID unique: {df['ClaimID'].nunique()}/{len(df)}"
        })
    
    # Not null checks
    not_null_columns = ['ClaimID', 'BeneID', 'Provider', 'ClaimStartDt', 'ClaimEndDt','AdmissionDt','DischargeDt']
    for col in not_null_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            success = null_count == 0
            results.append({
                'test': 'not_null',
                'column': col,
                'success': success,
                'message': f"'{col}' nulls: {null_count}"
            })
    
    # Positive claim amount
    if 'InscClaimAmtReimbursed' in df.columns:
        negative_count = (df['InscClaimAmtReimbursed'] < 0).sum()
        success = negative_count == 0
        results.append({
            'test': 'positive_value',
            'column': 'InscClaimAmtReimbursed',
            'success': success,
            'message': f"Negative amounts: {negative_count}"
        })
    
    # Row count
    results.append({
        'test': 'row_count',
        'column': None,
        'success': len(df) > 0,
        'message': f"Row count: {len(df)}"
    })
    
    return _build_report('inpatient', df, results, raise_on_fail)


# =============================================================================
# OUTPATIENT DATA VALIDATION
# =============================================================================

def validate_outpatient_data(df: pd.DataFrame, raise_on_fail: bool = False) -> Dict:
    """
    Validate outpatient claims data.
    
    Args:
        df: Outpatient claims DataFrame
        
    Returns:
        dict: Validation report with success status and details
    """
    results = []
    
    # Required columns
    required_columns = [
        'BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
       'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
       'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
       'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
       'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid',
       'ClmAdmitDiagnosisCode'
    ]
    
    for col in required_columns:
        success = col in df.columns
        results.append({
            'test': 'column_exists',
            'column': col,
            'success': success,
            'message': f"Column '{col}' exists" if success else f"Missing column: {col}"
        })
    
    # ClaimID uniqueness
    if 'ClaimID' in df.columns:
        is_unique = df['ClaimID'].nunique() == len(df)
        results.append({
            'test': 'unique_values',
            'column': 'ClaimID',
            'success': is_unique,
            'message': f"ClaimID unique: {df['ClaimID'].nunique()}/{len(df)}"
        })
    
    # Not null checks
    not_null_columns = ['ClaimID', 'BeneID', 'Provider', 'ClaimStartDt', 'ClaimEndDt']
    for col in not_null_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            success = null_count == 0
            results.append({
                'test': 'not_null',
                'column': col,
                'success': success,
                'message': f"'{col}' nulls: {null_count}"
            })
    
    # Positive claim amount
    if 'InscClaimAmtReimbursed' in df.columns:
        negative_count = (df['InscClaimAmtReimbursed'] < 0).sum()
        success = negative_count == 0
        results.append({
            'test': 'positive_value',
            'column': 'InscClaimAmtReimbursed',
            'success': success,
            'message': f"Negative amounts: {negative_count}"
        })
    
    # Row count
    results.append({
        'test': 'row_count',
        'column': None,
        'success': len(df) > 0,
        'message': f"Row count: {len(df)}"
    })
    
    return _build_report('outpatient', df, results, raise_on_fail)


# =============================================================================
# BENEFICIARY DATA VALIDATION
# =============================================================================

def validate_beneficiary_data(df: pd.DataFrame, raise_on_fail: bool = False) -> Dict:
    """
    Validate beneficiary data.
    
    Args:
        df: Beneficiary DataFrame
        
    Returns:
        dict: Validation report with success status and details
    """
    results = []
    
    # Required columns
    required_columns = [
        'BeneID', 'DOB', 'DOD', 'Gender', 'Race', 'RenalDiseaseIndicator',
       'State', 'County', 'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',
       'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
       'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
       'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',
       'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
       'ChronicCond_stroke', 'IPAnnualReimbursementAmt',
       'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt',
       'OPAnnualDeductibleAmt'
    ]
    
    for col in required_columns:
        success = col in df.columns
        results.append({
            'test': 'column_exists',
            'column': col,
            'success': success,
            'message': f"Column '{col}' exists" if success else f"Missing column: {col}"
        })
    
    # BeneID uniqueness
    if 'BeneID' in df.columns:
        is_unique = df['BeneID'].nunique() == len(df)
        results.append({
            'test': 'unique_values',
            'column': 'BeneID',
            'success': is_unique,
            'message': f"BeneID unique: {df['BeneID'].nunique()}/{len(df)}"
        })
    
    # Gender values (1=Male, 2=Female)
    if 'Gender' in df.columns:
        valid_genders = df['Gender'].isin([1, 2]).all()
        results.append({
            'test': 'valid_values',
            'column': 'Gender',
            'success': valid_genders,
            'message': f"Gender values in [1,2]: {valid_genders}"
        })
    
    # Race values (1-5)
    if 'Race' in df.columns:
        valid_races = df['Race'].isin([1, 2, 3, 4, 5]).all()
        results.append({
            'test': 'valid_values',
            'column': 'Race',
            'success': valid_races,
            'message': f"Race values in [1-5]: {valid_races}"
        })
    
    # RenalDiseaseIndicator values ('0', 'Y')
    if 'RenalDiseaseIndicator' in df.columns:
        valid_renal = df['RenalDiseaseIndicator'].isin(['0', 'Y', 0, 1]).all()
        results.append({
            'test': 'valid_values',
            'column': 'RenalDiseaseIndicator',
            'success': valid_renal,
            'message': f"RenalDiseaseIndicator valid: {valid_renal}"
        })
    
    # Not null checks
    not_null_columns = ['BeneID', 'DOB', 'Gender', 'Race']
    for col in not_null_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            success = null_count == 0
            results.append({
                'test': 'not_null',
                'column': col,
                'success': success,
                'message': f"'{col}' nulls: {null_count}"
            })
    
    # Row count
    results.append({
        'test': 'row_count',
        'column': None,
        'success': len(df) > 0,
        'message': f"Row count: {len(df)}"
    })
    
    return _build_report('beneficiary', df, results, raise_on_fail)


# =============================================================================
# INFERENCING DATA VALIDATION
# =============================================================================

def validate_inference_data(df: pd.DataFrame, raise_on_fail: bool = False) -> Dict:
    """
    Validate processed inference data (after feature engineering).
    
    Args:
        df: Inference DataFrame with engineered features
        
    Returns:
        dict: Validation report with success status and details
    """
    results = []
    
    # Required feature columns
    required_columns = [
       'count_unique_beneficiary',
       'count_unique_claims', 'count_dead_beneficiary', 'count_unique_states',
       'count_unique_counties', 'mean_hospital_stay_days',
       'max_hospital_stay_days', 'total_top_diagnosis_codes',
       'mean_total_diagnosis', 'mean_total_procedures',
       'mean_number_of_physicians', 'mean_difference_stay_vs_claim',
       'patients_under_top_attending_physician',
       'patients_under_top_operating_physician', 'mean_claim_amount',
       'total_claim_amount', 'std_claim_amount', 'mean_age', 'count_of_males',
       'count_of_females', 'count_of_black_people', 'count_of_white_people',
       'count_of_hispanic_people', 'count_of_other_people', 'count_alzheimer',
       'count_heartfailure', 'count_kidneydisease', 'count_cancer',
       'count_obstrpulmonary', 'count_depression', 'count_diabetes',
       'count_ischemicheart', 'count_osteoporasis',
       'count_rheumatoidarthritis', 'count_stroke', 'count_renal_disease',
       'mean_annual_inpatient_reimbursement',
       'mean_annual_inpatient_deductible',
       'mean_annual_outpatient_reimbursement',
       'mean_annual_outpatient_deductible', 'count_inpatient',
       'count_outpatient', 'claims_per_bene'
    ]
    
    for col in required_columns:
        success = col in df.columns
        results.append({
            'test': 'column_exists',
            'column': col,
            'success': success,
            'message': f"Column '{col}' exists" if success else f"Missing column: {col}"
        })
    
    # No nulls in features
    feature_cols = [c for c in df.columns if c not in required_columns]
    total_nulls = df[feature_cols].isnull().sum().sum()
    results.append({
        'test': 'no_nulls',
        'column': 'features',
        'success': total_nulls == 0,
        'message': f"Total nulls in features: {total_nulls}"
    })    
    # Row count
    results.append({
        'test': 'row_count',
        'column': None,
        'success': len(df) > 0,
        'message': f"Row count: {len(df)}"
    })
    
    # Feature count (should have 44+ features)
    feature_count = len(df.columns) - 2  # Exclude Provider, PotentialFraud
    results.append({
        'test': 'feature_count',
        'column': None,
        'success': feature_count >= 40,
        'message': f"Feature count: {feature_count} (expected >= 40)"
    })
    
    return _build_report('inference_data', df, results, raise_on_fail)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _build_report(dataset_name: str, df: pd.DataFrame, results: List[Dict], raise_on_fail: bool = False) -> Dict:
    """Build validation report from results."""
    all_success = all(r['success'] for r in results)
    failed = [r for r in results if not r['success']]
    
    report = {
        'dataset': dataset_name,
        'success': all_success,
        'total_checks': len(results),
        'passed': len([r for r in results if r['success']]),
        'failed': len(failed),
        'failed_tests': failed,
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    if not all_success and raise_on_fail:
        failed_msgs = [f"{t['test']} on {t.get('column', 'table')}" for t in failed]
        raise ValueError(f"Validation failed for {dataset_name}: {failed_msgs}")
    
    return report


