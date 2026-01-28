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

st.title("🎯 Smart Loan Approval System – Stacking Model")
st.markdown("""
This application uses a **Stacking Ensemble Machine Learning model**
to predict loan approval by combining multiple models.
""")

# -----------------------------------
# LOAD DATA (AUTO-DETECT SEPARATOR)
# -----------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("loan.csv")          # try comma
        if len(df.columns) == 1:
            df = pd.read_csv("loan.csv", sep="\t")  # fallback to tab
    except Exception:
        df = pd.read_csv("loan.csv", sep="\t")

    df.columns = df.columns.str.strip()
    return df

data = load_data()

# -----------------------------------
# SHOW COLUMNS (DEBUG – SAFE TO KEEP)
# -----------------------------------
st.write("📄 Dataset Columns:", data.columns.tolist())

# -----------------------------------
# VALIDATE REQUIRED COLUMNS
# -----------------------------------
required_columns = [
    "Loan_ID", "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
    "LoanAmount", "Loan_Amount_Term", "Credit_History",
    "Property_Area", "Loan_Status"
]

missing = [c for c in required_columns if c not in data.columns]
if missing:
    st.error(f"❌ Missing columns in dataset: {missing}")
    st.stop()

# -----------------------------------
# HANDLE MISSING VALUES
# -----------------------------------
data["LoanAmount"] = pd.to_numeric(data["LoanAmount"], errors="coerce")
data["LoanAmount"].fillna(data["LoanAmount"].median(), inplace=True)

data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].median(), inplace=True)
data["Credit_History"].fillna(1, inplace=True)

# -----------------------------------
# DROP ID COLUMN
# -----------------------------------
data.drop("Loan_ID", axis=1, inplace=True)

# -----------------------------------
# ENCODE CATEGORICAL FEATURES
# -----------------------------------
le = LabelEncoder()
for col in ["Gender", "Married", "Dependents", "Education",
            "Self_Employed", "Property_Area", "Loan_Status"]:
    data[col] = le.fit_transform(data[col])

# -----------------------------------
# FEATURES & TARGET
# -----------------------------------
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# -----------------------------------
# TRAIN–TEST SPLIT
# -----------------------------------
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

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

app_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapp_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term", value=360)
credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])

input_data = pd.DataFrame([{
    "Gender": 1 if gender == "Male" else 0,
    "Married": 1 if married == "Yes" else 0,
    "Dependents": 3 if dependents == "3+" else int(dependents),
    "Education": 1 if education == "Graduate" else 0,
    "Self_Employed": 1 if self_employed == "Yes" else 0,
    "ApplicantIncome": app_income,
    "CoapplicantIncome": coapp_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": 1 if credit_history == "Yes" else 0,
    "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
}])

# -----------------------------------
# MODEL ARCHITECTURE DISPLAY
# -----------------------------------
st.subheader("🧩 Stacking Model Architecture")
st.markdown("""
**Base Models Used**
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Meta Model Used**
- Logistic Regression
""")

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    pred_lr = lr.predict(input_data)[0]
    pred_dt = dt.predict(input_data)[0]
    pred_rf = rf.predict(input_data)[0]

    meta_input = np.array([[pred_lr, pred_dt, pred_rf]])
    final_pred = meta_model.predict(meta_input)[0]
    confidence = np.max(meta_model.predict_proba(meta_input)) * 100

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

    st.subheader("💼 Business Explanation")
    if final_pred == 1:
        st.info(
            "Based on income, credit history, and combined predictions from multiple "
            "models, the applicant is likely to repay the loan. Therefore, the stacking "
            "model predicts loan approval."
        )
    else:
        st.info(
            "Based on income level, credit history, and combined model predictions, "
            "the applicant may face difficulty in repayment. Therefore, the stacking "
            "model predicts loan rejection."
        )
