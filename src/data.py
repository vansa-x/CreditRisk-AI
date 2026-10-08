import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(filepath, target_col='MIS_Status'):
    """
    Ingests raw credit data, handles missing values, scales features,
    and applies SMOTE strictly to the training set to prevent data leakage.
    """
    # 1. Ingestion
    df = pd.read_csv(filepath)
    
    # 2. Feature/Target Isolation
    X = df.drop(columns=[target_col])
    y = df[target_col] 

    # 3. Train-Test Split (Executed BEFORE any transformations)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Imputation
    num_imputer = SimpleImputer(strategy='median')
    X_train_imputed = num_imputer.fit_transform(X_train)
    X_test_imputed = num_imputer.transform(X_test)

    # 5. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # 6. Class Imbalance Handling (SMOTE)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, num_imputer, scaler