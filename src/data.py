import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(filepath, target_col='MIS_Status'):
    """
    Ingests the SBA dataset, cleans financial features, handles missing values, 
    scales features, and applies SMOTE strictly to the training set.
    """
    # 1. Ingestion & Filtering (Added low_memory=False to fix the DtypeWarning)
    df = pd.read_csv(filepath, low_memory=False)
    keep_cols = ['Term', 'GrAppv', 'SBA_Appv', 'NoEmp', 'RetainedJob', 'NewExist', target_col]
    df = df[keep_cols]

    # 2. Target Cleaning & Mapping (Fixed to handle messy Kaggle text)
    df = df.dropna(subset=[target_col])
    # Strip any hidden spaces from the text
    df[target_col] = df[target_col].astype(str).str.strip()
    # Replace variations of the text with binary integers
    df[target_col] = df[target_col].replace({'P I F': 0, 'PIF': 0, 'CHGOFF': 1})
    
    # Keep ONLY rows that successfully mapped to 0 or 1, then force integer type
    df = df[df[target_col].isin([0, 1])]
    df[target_col] = df[target_col].astype(int)

    # 3. Currency Formatting
    # Strips '$' and ',' from financial columns and converts to numeric floats
    for col in ['GrAppv', 'SBA_Appv']:
        df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).astype(float)

    # 4. Feature/Target Isolation
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 5. Train-Test Split (Executed BEFORE any transformations to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Imputation (Median prevents extreme financial outliers from skewing the baseline)
    num_imputer = SimpleImputer(strategy='median')
    X_train_imputed = num_imputer.fit_transform(X_train)
    X_test_imputed = num_imputer.transform(X_test)

    # 7. Scaling (Ensures preprocessing and scaling are executed properly for the pipeline)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # 8. Class Imbalance Handling (SMOTE)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, num_imputer, scaler