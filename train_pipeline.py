import joblib
from src.data import load_and_preprocess_data
from src.model import train_models

def run_pipeline():
    """
    Executes the end-to-end data processing and training pipeline,
    then serializes the artifacts for production inference.
    """
    print("Initializing pipeline...")
    data_path = 'src/sba_data.csv'
    
    # 1. Load and Preprocess Data
    # The function handles train/test split, imputation, scaling, and SMOTE
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, imputer, scaler = load_and_preprocess_data(data_path)
    
    # 2. Train Models (Robust Cross-Validation & Hyperparameter Tuning)
    print("Training models...")
    xgb_best = train_models(X_train, y_train)
    
    # 3. Serialize Artifacts
    print("Saving production artifacts...")
    # Saving the XGBoost model as requested for the underwriting engine
    joblib.dump(xgb_best, 'xgb_model.pkl')
    # The scaler must be saved to apply the exact same transformation to live inference data
    joblib.dump(scaler, 'scaler.pkl') 
    
    print("Pipeline complete. Artifacts saved as .pkl files.")

if __name__ == "__main__":
    run_pipeline()