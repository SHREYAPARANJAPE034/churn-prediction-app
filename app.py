import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# ----------------------------

# Load Model & Encoders

# ----------------------------

with open("models/model.pkl", "rb") as f:
model = pickle.load(f)

with open("models/encoders.pkl", "rb") as f:
encoders = pickle.load(f)

# ----------------------------

# UI

# ----------------------------

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# ----------------------------

# Inputs

# ----------------------------

gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
"Payment Method",
[
"Electronic check",
"Mailed check",
"Bank transfer (automatic)",
"Credit card (automatic)"
]
)

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)

# ----------------------------

# Prediction

# ----------------------------

if st.button("Predict"):


input_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}])

# Encode categorical columns
for col in encoders:
    input_data[col] = encoders[col].transform(input_data[col])

# Prediction
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

st.subheader("Prediction Result")

if prediction == 1:
    st.error(f"Customer is likely to churn 😟 (Probability: {probability:.2f})")
else:
    st.success(f"Customer is not likely to churn 😊 (Probability: {probability:.2f})")

# ----------------------------
# SHAP Explainability
# ----------------------------
st.subheader("🔍 Why this prediction? (SHAP Explainability)")

explainer = shap.Explainer(model)
shap_values = explainer(input_data)

fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)
