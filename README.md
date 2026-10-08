```markdown
# CreditRisk AI: Commercial Underwriting Engine 🏦📈

**Applied AI & Machine Learning Capstone Project**

CreditRisk AI is an end-to-end machine learning pipeline and interactive executive dashboard designed to predict commercial loan defaults. Built using the U.S. Small Business Administration (SBA) National dataset, this tool translates complex XGBoost predictions into Expected Value (EV) metrics and actionable business recommendations for a Chief Risk Officer (CRO).

## 🚀 Features
* **Real-Time Inference:** Live prediction of loan default probabilities based on user-adjusted financial parameters.
* **Executive ROI Dashboard:** Automatically calculates Expected Revenue, Expected Loss, and Net Expected Value to generate an Approve/Reject recommendation.
* **Regulatory Explainability (SHAP):** Generates real-time waterfall charts to explain exactly which features influenced the AI's decision, ensuring compliance with underwriting audit trails.

## 📂 Repository Structure
This project follows a clean, modular architecture:
* `app.py`: The Streamlit web application and executive dashboard UI.
* `src/data.py`: Handles raw data ingestion, median imputation, StandardScaler formatting, and SMOTE class balancing.
* `src/model.py`: Contains robust hyperparameter tuning and model training logic (RandomizedSearchCV), plus business ROI calculations.
* `train_pipeline.py`: The execution script that runs the pipeline and serializes the trained artifacts.
* `requirements.txt`: Project dependencies.
*(Note: The raw 170MB Kaggle dataset is excluded via `.gitignore` to adhere to best practices).*

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/vansa-x/CreditRisk-AI.git](https://github.com/vansa-x/CreditRisk-AI.git)
   cd CreditRisk-AI

```

2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the Streamlit Dashboard:**
```bash
python -m streamlit run app.py

```



## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Machine Learning:** XGBoost, Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Explainable AI:** SHAP
* **Data Processing:** Pandas, NumPy
