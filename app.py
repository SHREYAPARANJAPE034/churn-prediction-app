import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.predict import predict, explainer

st.set_page_config(page_title="Churn Predictor", layout="wide")

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>📊 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>ML + Explainable AI Dashboard</p>", unsafe_allow_html=True)

st.divider()

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/churn.csv")

# ---------------- EDA SECTION ----------------
st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(4,3))
    sns.countplot(x='Churn', data=df, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(4,3))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# ---------------- INPUT FORM ----------------
st.subheader("🧾 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72)

with col2:
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.divider()

col3, col4 = st.columns(2)

with col3:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col4:
    monthly = st.number_input("Monthly Charges", 0.0)
    total = st.number_input("Total Charges", 0.0)

# ---------------- VALIDATION ----------------
expected_total = tenure * monthly

st.info(f"💡 Expected Total Charges ≈ {expected_total:.2f}")

is_valid = True
if abs(total - expected_total) > 100:
    st.warning("⚠️ Total Charges seems inconsistent with Tenure × Monthly Charges")
    is_valid = False

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Churn", use_container_width=True) and is_valid:

    input_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    pred, prob, shap_values, df_input = predict(input_data)

    st.subheader("📈 Prediction Result")

    st.progress(float(prob))

    if pred == 1:
        st.error(f"⚠️ High Churn Risk ({prob:.2f})")
    else:
        st.success(f"✅ Customer Likely to Stay ({prob:.2f})")

    # ---------------- SHAP ----------------
    # ---------------- SHAP ----------------
    # ---------------- SHAP ----------------
st.subheader("🔍 Why this prediction? (SHAP Explainability)")

fig, ax = plt.subplots()

shap_exp = shap.Explanation(
    values=shap_values[0],              # SHAP values
    base_values=explainer.expected_value,
    data=df_input.iloc[0],
    feature_names=df_input.columns.tolist()
)

shap.plots.waterfall(shap_exp, show=False)

st.pyplot(fig)

    

    
      

    

# python -m streamlit run app.py