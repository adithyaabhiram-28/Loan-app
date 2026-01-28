import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Smart Loan Approval System – Stacking Model",
    layout="wide"
)

# -----------------------------------
# TITLE & DESCRIPTION
# -----------------------------------
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.markdown(
    """
    **This system uses a Stacking Ensemble Machine Learning model** to predict  
    whether a loan will be approved by combining multiple ML models for  
    better and more reliable decision making.
    """
)

# -----------------------------------
# LOAD & PREPROCESS DATA
# -----------------------------------
data = pd.read_csv("loan.csv")

# Encode categorical features
le = LabelEncoder()
for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]:
    data[col] = le.fit_transform(data[col])

data = data.dropna()

X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# BASE MODELS
# -----------------------------------
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# -----------------------------------
# META MODEL (STACKING)
# -----------------------------------
train_meta = np.column_stack([
    lr.predict(X_train),
    dt.predict(X_train),
    rf.predict(X_train)
])

meta_model = LogisticRegression()
meta_model.fit(train_meta, y_train)

# -----------------------------------
# SIDEBAR INPUTS
# -----------------------------------
st.sidebar.header("📥 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapp_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term", min_value=0)
credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

credit_history = 1 if credit_history == "Yes" else 0
employment = 0 if employment == "Salaried" else 1
property_area = {"Urban": 2, "Semi-Urban": 1, "Rural": 0}[property_area]

# -----------------------------------
# MODEL ARCHITECTURE DISPLAY
# -----------------------------------
st.subheader("🧩 Stacking Model Architecture")

st.markdown("""
**Base Models Used:**
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Meta Model Used:**
- Logistic Regression  

📌 *Predictions from base models are combined and passed to the meta-model to make the final decision.*
""")

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    input_data = np.array([[
        1, 1, 1, employment, app_income, coapp_income,
        loan_amount, loan_term, credit_history, property_area
    ]])

    # Base model predictions
    pred_lr = lr.predict(input_data)[0]
    pred_dt = dt.predict(input_data)[0]
    pred_rf = rf.predict(input_data)[0]

    # Meta model prediction
    meta_input = np.array([[pred_lr, pred_dt, pred_rf]])
    final_pred = meta_model.predict(meta_input)[0]
    confidence = np.max(meta_model.predict_proba(meta_input)) * 100

    # -----------------------------------
    # OUTPUT SECTION
    # -----------------------------------
    st.subheader("📊 Prediction Result")

    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.markdown("### 📊 Base Model Predictions")
    st.write(f"**Logistic Regression:** {'Approved' if pred_lr == 1 else 'Rejected'}")
    st.write(f"**Decision Tree:** {'Approved' if pred_dt == 1 else 'Rejected'}")
    st.write(f"**Random Forest:** {'Approved' if pred_rf == 1 else 'Rejected'}")

    st.markdown("### 🧠 Final Stacking Decision")
    st.write("**Approved**" if final_pred == 1 else "**Rejected**")

    st.markdown(f"### 📈 Confidence Score")
    st.write(f"{confidence:.2f}%")

    # -----------------------------------
    # BUSINESS EXPLANATION (MANDATORY)
    # -----------------------------------
    st.subheader("💼 Business Explanation")

    if final_pred == 1:
        st.info(
            "Based on the applicant’s income, credit history, and combined predictions "
            "from multiple machine learning models, the applicant is likely to repay "
            "the loan. Therefore, the stacking model predicts **loan approval**."
        )
    else:
        st.info(
            "Based on income levels, credit history, and combined model predictions, "
            "the applicant may face difficulty in repaying the loan. Therefore, "
            "the stacking model predicts **loan rejection**."
        )
