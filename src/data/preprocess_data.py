import pandas as pd 
import numpy as np 
import os

def process_beneficiary_data(df) -> pd.DataFrame:
    """
    - ChronicCond columns: 1/2 → 0/1
    - Gender: one-hot encode
    - Race: one-hot encode
    - RenalDiseaseIndicator: '0'/'Y' → 0/1
    """

    df.loc[:, df.columns.str.contains('ChronicCond_')] = \
    df.filter(like='ChronicCond_').replace({1: 1, 2: 0})

    df['RenalDiseaseIndicator'] = df['RenalDiseaseIndicator'].map({'0':0,'Y':1})

    df = pd.concat([df,
        pd.get_dummies(df['Gender'].map({1: 'Male', 2: 'Female'}), prefix='Gender'),
        pd.get_dummies(df['Race'].map({1: 'White', 2: 'Black', 3: 'Other', 4: 'Unknown', 5: 'Hispanic'}), prefix='Race')], axis=1)

    return df

def process_claims_data(in_df,out_df) -> pd.DataFrame:
    """
    - Clean and process combined claims data from inpatient and outpatient
    """

    in_df['Is_Inpatient'] = True
    out_df['Is_Outpatient'] = True

    df = pd.concat([in_df,out_df])

    for i in ['ClaimStartDt','ClaimEndDt','AdmissionDt','DischargeDt']:
        df[i] = pd.to_datetime(df[i])

    return df

def process_training_data(df) -> pd.DataFrame:
    """
    - Fill missing values for prepared data before training
    """
    df['std_claim_amount'] = df['std_claim_amount'].fillna(0)
    df['PotentialFraud'] = df['PotentialFraud'].map({"No": 0, "Yes": 1})

    return df

def process_inference_data(df) -> pd.DataFrame:

    df['std_claim_amount'] = df['std_claim_amount'].fillna(0)
    df = df.drop(['Provider','PotentialFraud'],axis=1)

    return df