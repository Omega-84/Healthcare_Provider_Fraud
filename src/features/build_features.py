import os 
import pandas as pd 
import numpy as np 

def generate_top_codes(df, save_directory) -> None:
    top_diagnosis_code = df.groupby(['ClmAdmitDiagnosisCode'])['ClaimID'].count().reset_index().sort_values(by='ClaimID', ascending=False).head(20)['ClmAdmitDiagnosisCode'].unique()
    pd.DataFrame(top_diagnosis_code,columns=['ClmAdmitDiagnosisCode']).to_csv(os.path.join(save_directory,"top_diagnosis_code.csv"),index=False)
    top_20_attending = df['AttendingPhysician'].value_counts().head(20).index
    top_20_operating = df['OperatingPhysician'].value_counts().head(20).index
    pd.DataFrame(top_20_attending,columns=['AttendingPhysician']).to_csv(os.path.join(save_directory,"top_attending_physician.csv"),index=False)
    pd.DataFrame(top_20_operating,columns=['OperatingPhysician']).to_csv(os.path.join(save_directory,"top_operating_physician.csv"),index=False)


def create_claims_features(df: pd.DataFrame,top_diagnosis_code: pd.DataFrame,top_attending_physician: pd.DataFrame,top_operating_physician: pd.DataFrame) -> pd.DataFrame:
    df['billing_before_admission'] = (df['ClaimStartDt'] - df['AdmissionDt']).dt.days < 0            
    df['billing_after_discharge'] = (df['ClaimEndDt'] - df['DischargeDt']).dt.days > 0
    df['top_diagnosis_code'] = np.where(df['ClmAdmitDiagnosisCode'].isin(top_diagnosis_code['ClmAdmitDiagnosisCode']),1,0)
    df['hospital_stay_days'] = (df['DischargeDt'] - df['AdmissionDt']).dt.days + 1
    df['num_diagnoses'] =df.filter(like='ClmDiagnosisCode_').notna().sum(axis=1)
    df['num_procedures'] =df.filter(like='ClmProcedureCode_').notna().sum(axis=1)
    df['claim_duration_days'] = (df['ClaimEndDt'] - df['ClaimStartDt']).dt.days + 1
    df['num_of_physicians'] = df.filter(like='Physician').notna().sum(axis=1)
    df['is_top20_attending'] = np.where(df['AttendingPhysician'].isin(top_attending_physician['AttendingPhysician']),1,0)
    df['is_top20_operating'] = np.where(df['OperatingPhysician'].isin(top_operating_physician['OperatingPhysician']),1,0)
    df['stay_vs_claim_diff'] = df['hospital_stay_days'] - df['claim_duration_days']

    return df

def merge_beneficiary_and_claims(beneficiary_df: pd.DataFrame,claims_df: pd.DataFrame) -> pd.DataFrame:
    
    df = claims_df.merge(beneficiary_df,on='BeneID',how='left')
    df['DOB'] = pd.to_datetime(df['DOB'])
    df['Age_at_Claim'] = ((pd.to_datetime(df['ClaimStartDt']) - pd.to_datetime(df['DOB'])).dt.days / 365.25).astype(int)
    df['Is_Dead'] = np.where(df['DOD'].isna(), False, True)

    return df

def aggregate_claims_features(df: pd.DataFrame) -> pd.DataFrame:

    for i in ['hospital_stay_days','stay_vs_claim_diff']:
        df[i] = df[i].fillna(0)
    for i in ['Is_Inpatient','Is_Outpatient']:
        df[i] = df[i].fillna(False)
    
    features = df.groupby(['Provider']).agg(
    count_unique_beneficiary = ('BeneID','nunique'),
    count_unique_claims = ('ClaimID','count'),
    count_dead_beneficiary = ('Is_Dead','sum'),
    count_unique_states = ('State','nunique'),
    count_unique_counties = ('County','nunique'),
    mean_hospital_stay_days = ('hospital_stay_days', 'mean'),
    max_hospital_stay_days = ('hospital_stay_days', 'max'),
    total_top_diagnosis_codes = ('top_diagnosis_code','sum'),
    mean_total_diagnosis =('num_diagnoses','mean'),
    mean_total_procedures = ('num_procedures','mean'),
    mean_number_of_physicians = ('num_of_physicians','mean'),
    mean_difference_stay_vs_claim = ('stay_vs_claim_diff','mean'),
    patients_under_top_attending_physician = ('is_top20_attending','sum'),
    patients_under_top_operating_physician = ('is_top20_operating','sum'),
    mean_claim_amount = ('InscClaimAmtReimbursed','mean'),
    total_claim_amount = ('InscClaimAmtReimbursed','sum'),
    std_claim_amount = ('InscClaimAmtReimbursed','std'),
    mean_age = ('Age_at_Claim','mean'),
    count_of_males = ('Gender_Male','sum'),
    count_of_females = ('Gender_Female','sum'),
    count_of_black_people = ('Race_Black','sum'),
    count_of_white_people = ('Race_White','sum'),
    count_of_hispanic_people = ('Race_Hispanic','sum'),
    count_of_other_people = ('Race_Other','sum'),
    count_alzheimer = ('ChronicCond_Alzheimer', 'sum'),
    count_heartfailure = ('ChronicCond_Heartfailure', 'sum'),
    count_kidneydisease = ('ChronicCond_KidneyDisease', 'sum'),
    count_cancer = ('ChronicCond_Cancer', 'sum'),
    count_obstrpulmonary = ('ChronicCond_ObstrPulmonary', 'sum'),
    count_depression = ('ChronicCond_Depression', 'sum'),
    count_diabetes = ('ChronicCond_Diabetes', 'sum'),
    count_ischemicheart = ('ChronicCond_IschemicHeart', 'sum'),
    count_osteoporasis = ('ChronicCond_Osteoporasis', 'sum'),
    count_rheumatoidarthritis = ('ChronicCond_rheumatoidarthritis', 'sum'),
    count_stroke = ('ChronicCond_stroke', 'sum'), 
    count_renal_disease = ('RenalDiseaseIndicator', 'sum'),   
    mean_annual_inpatient_reimbursement = ('IPAnnualReimbursementAmt','mean'),
    mean_annual_inpatient_deductible = ('IPAnnualDeductibleAmt','mean'),
    mean_annual_outpatient_reimbursement = ('OPAnnualReimbursementAmt','mean'),
    mean_annual_outpatient_deductible = ('OPAnnualDeductibleAmt','mean'),
    count_inpatient = ('Is_Inpatient','sum'),
    count_outpatient = ('Is_Outpatient','sum')
        ).reset_index()

    features['claims_per_bene'] = features['count_unique_claims'] / features['count_unique_beneficiary']
    
    return features

def merge_provider_train(features: pd.DataFrame, provider: pd.DataFrame) -> pd.DataFrame:
    
    df = provider.merge(features,on='Provider',how='left')
    df['std_claim_amount'] = df['std_claim_amount'].fillna(0)
    df['PotentialFraud'] = df['PotentialFraud'].map({"No": 0, "Yes": 1})

    return df
