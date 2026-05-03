import pickle
import pandas as pd
import shap

# Load model & encoders
model = pickle.load(open("models/model.pkl", "rb"))
encoders = pickle.load(open("models/encoders.pkl", "rb"))

# Feature order (VERY IMPORTANT)
FEATURE_ORDER = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# SHAP explainer (TreeExplainer works best for XGBoost)
explainer = shap.TreeExplainer(model)

def predict(data_dict):
    df = pd.DataFrame([data_dict])

    # Ensure all columns exist
    for col in FEATURE_ORDER:
        if col not in df:
            df[col] = 0

    # Correct order
    df = df[FEATURE_ORDER]

    # Encode safely
    for col, le in encoders.items():
        if col in df:
            try:
                df[col] = le.transform(df[col])
            except:
                df[col] = 0

    df = df.astype(float)

    # Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    # SHAP values
    shap_values = explainer.shap_values(df)

    return prediction, probability, shap_values, df