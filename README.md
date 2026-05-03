# 📊 Customer Churn Prediction System

## 🔍 Overview
This project is an end-to-end Machine Learning application that predicts whether a customer is likely to churn (leave the service) based on their demographic, service usage, and billing information.

The system includes:
- Data preprocessing
- Model training using XGBoost
- Interactive Streamlit web application
- SHAP-based explainability
- Deployment-ready structure

---

## 🎯 Objective
To help businesses identify high-risk customers and take proactive actions to improve retention.

---

## 🧠 Machine Learning Model
- Algorithm: XGBoost (Extreme Gradient Boosting)
- Problem Type: Binary Classification
- Output:
  - Churn Prediction (Yes/No)
  - Probability Score

---

## 📊 Features Used
- Customer Info: Gender, Senior Citizen, Partner, Dependents
- Services: Internet, Streaming, Security, Support
- Billing: Contract Type, Payment Method, Charges
- Tenure & Usage Patterns

---

## 📈 Key Features of the Project

### ✅ Interactive Dashboard
Built using Streamlit with a clean UI and real-time predictions.

### 📊 EDA Visualizations
- Churn distribution
- Monthly charges vs churn
- Contract type analysis

### 🔍 SHAP Explainability
Explains why a prediction was made by showing feature contributions.

### ⚠️ Input Validation
Ensures consistency between:
Total Charges ≈ Tenure × Monthly Charges

### 💡 Business Insights
Provides actionable suggestions based on churn probability.

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib, Seaborn
- SHAP

---

## 📂 Project Structure
churn-prediction/
│── data/
│   └── churn.csv
│
│── models/
│   ├── model.pkl
│   └── encoders.pkl
│
│── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
│── app.py
│── requirements.txt
│── README.md

---

## 🚀 How to Run Locally

1. Clone the repository:
git clone https://github.com/your-username/churn-prediction-app.git
cd churn-prediction-app

2. Install dependencies:
pip install -r requirements.txt

3. Run the application:
python -m streamlit run app.py

---

## 🌐 Deployment
The application can be deployed using Streamlit Cloud for free.

---

## 📌 Sample Use Case
- Input customer details
- Get churn prediction instantly
- Understand key factors affecting churn
- Take action (e.g., offer discounts, improve service)

---

## 💼 Resume Highlight
Developed an end-to-end customer churn prediction system using XGBoost with an interactive Streamlit dashboard, including EDA visualizations and SHAP-based explainability, deployed on cloud.

---

## 🔮 Future Improvements
- Add authentication system
- Store prediction history
- Integrate real-time APIs
- Improve model with deep learning

---

## 👩‍💻 Author
Shreya Paranjape

---

## ⭐ If you like this project
Give it a star on GitHub!