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
# TITLE
# -----------------------------------
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.markdown("""
This application uses a **Stacking Ensemble Machine Learning model**
to predict loan approval by combining multiple models for better decision making.
""")

# -----------------------------------
# LOAD DATA
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("loan.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

data = load_data()
data = data.dropna()

# -----------------------------------
# IDENTIFY TARGET COLUMN SAFELY
# -----------------------------------
possible_targets = [
    "loan_status", "loanstatus", "status", "target", "approved"
]

target_col = None
for col in data.columns:
    if col in possible_targets:
        target_col = col
        break

if target_col is None:
    st.error("❌ Target column not found in dataset.")
    st.stop()

# -----------------------------------
# ENCODING
# -----------------------------------
le = LabelEncoder()
categorical_cols = data.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# -----------------------------------
# FEATURES & LABEL
# -----------------------------------
X = data.drop(target_col, axis=1)
y = data[target_col]

# -----------------------------------
# TRAIN TEST SPLIT
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

user_inputs = {}
for col in X.columns:
    user_inputs[col] = st.sidebar.number_input(col.replace("_", " ").title(), min_value=0.0)

input_df = pd.DataFrame([user_inputs])

# -----------------------------------
# MODEL ARCHITECTURE
# -----------------------------------
st.subheader("🧩 Stacking Model Architecture")
st.markdown("""
**Base Models Used**
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Meta Model**
- Logistic Regression  

📌 Base model predictions are used as inputs to the meta-model.
""")

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    pred_lr = lr.predict(input_df)[0]
    pred_dt = dt.predict(input_df)[0]
    pred_rf = rf.predict(input_df)[0]

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
            "Based on the applicant’s financial details and the combined predictions "
            "from multiple machine learning models, the applicant is likely to repay "
            "the loan. Therefore, the stacking model predicts **loan approval**."
        )
    else:
        st.info(
            "Based on income-related factors and combined model predictions, the "
            "applicant may face difficulty in loan repayment. Therefore, the stacking "
            "model predicts **loan rejection**."
        )
