import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
# LOAD DATA
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("loan.csv")
    df.columns = df.columns.str.strip()  # remove hidden spaces
    return df

data = load_data()

# -----------------------------------
# PREPROCESS DATA
# -----------------------------------
data = data.dropna()

# Encode categorical columns safely
le = LabelEncoder()
categorical_cols = data.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Separate features and target
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# BASE MODELS
# -----------------------------------
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# -----------------------------------
# STACKING META MODEL
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

📌 Predictions from base models are combined and passed to the meta-model for final decision making.
""")

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    # Create input dataframe aligned with training data
    input_data = pd.DataFrame([{
        "Gender": 1,
        "Married": 1,
        "Dependents": 0,
        "Education": 1,
        "Self_Employed": employment,
        "ApplicantIncome": app_income,
        "CoapplicantIncome": coapp_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    # Align input columns exactly with model training columns
    input_data = input_data[X.columns]

    # Base predictions
    pred_lr = lr.predict(input_data)[0]
    pred_dt = dt.predict(input_data)[0]
    pred_rf = rf.predict(input_data)[0]

    # Stacking prediction
    meta_input = np.array([[pred_lr, pred_dt, pred_rf]])
    final_pred = meta_model.predict(meta_input)[0]
    confidence = np.max(meta_model.predict_proba(meta_input)) * 100

    # -----------------------------------
    # OUTPUT
    # -----------------------------------
    st.subheader("📊 Prediction Result")

    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.markdown("### 📊 Base Model Predictions")
    st.write(f"Logistic Regression → {'Approved' if pred_lr else 'Rejected'}")
    st.write(f"Decision Tree → {'Approved' if pred_dt else 'Rejected'}")
    st.write(f"Random Forest → {'Approved' if pred_rf else 'Rejected'}")

    st.markdown("### 🧠 Final Stacking Decision")
    st.write("**Approved**" if final_pred else "**Rejected**")

    st.markdown("### 📈 Confidence Score")
    st.write(f"{confidence:.2f}%")

    # -----------------------------------
    # BUSINESS EXPLANATION (MANDATORY)
    # -----------------------------------
    st.subheader("💼 Business Explanation")

    if final_pred == 1:
        st.info(
            "Based on the applicant’s income, credit history, and the combined "
            "predictions from multiple machine learning models, the applicant is "
            "likely to repay the loan. Therefore, the stacking model predicts "
            "**loan approval**."
        )
    else:
        st.info(
            "Based on income levels, credit history, and combined predictions from "
            "multiple models, the applicant may face difficulty in repaying the loan. "
            "Therefore, the stacking model predicts **loan rejection**."
        )
