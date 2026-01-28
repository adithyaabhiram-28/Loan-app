<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Smart Loan Approval System – Stacking Model</title>
    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            background-color: #0f172a;
            color: #e5e7eb;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 1000px;
            margin: auto;
            padding: 40px;
        }
        h1, h2, h3 {
            color: #38bdf8;
        }
        p {
            line-height: 1.6;
        }
        ul {
            margin-left: 20px;
        }
        code {
            background-color: #1e293b;
            padding: 4px 8px;
            border-radius: 4px;
            color: #a5f3fc;
        }
        .box {
            background-color: #020617;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #38bdf8;
        }
        .success {
            border-left-color: #22c55e;
        }
        .warning {
            border-left-color: #f97316;
        }
        footer {
            text-align: center;
            margin-top: 40px;
            color: #94a3b8;
        }
    </style>
</head>
<body>

<div class="container">

    <h1>🎯 Smart Loan Approval System – Stacking Model</h1>

    <p>
        The <strong>Smart Loan Approval System</strong> is a Machine Learning web application
        that predicts whether a loan will be approved or rejected using a
        <strong>Stacking Ensemble Learning technique</strong>.
        Multiple machine learning models are combined to improve prediction accuracy
        and decision reliability.
    </p>

    <div class="box">
        <h2>📌 Project Objectives</h2>
        <ul>
            <li>Predict loan approval based on applicant financial details</li>
            <li>Demonstrate the working of a Stacking Ensemble Model</li>
            <li>Provide a business-friendly explanation of predictions</li>
            <li>Create an interactive and user-friendly web application</li>
        </ul>
    </div>

    <h2>🧠 Machine Learning Approach</h2>

    <div class="box">
        <h3>Base Models Used</h3>
        <ul>
            <li>Logistic Regression</li>
            <li>Decision Tree Classifier</li>
            <li>Random Forest Classifier</li>
        </ul>

        <h3>Meta Model</h3>
        <ul>
            <li>Logistic Regression</li>
        </ul>

        <p>
            Predictions from all base models are used as inputs to the meta-model,
            which produces the final loan approval decision.
        </p>
    </div>

    <h2>📊 Dataset Information</h2>

    <div class="box">
        <p>
            The system uses a loan dataset containing applicant demographic and
            financial information. The target variable is:
        </p>

        <code>Loan_Status</code>

        <p>
            Where:
        </p>
        <ul>
            <li><strong>Y</strong> → Loan Approved</li>
            <li><strong>N</strong> → Loan Rejected</li>
        </ul>

        <p>Key features include:</p>
        <ul>
            <li>Applicant Income</li>
            <li>Co-applicant Income</li>
            <li>Loan Amount</li>
            <li>Credit History</li>
            <li>Employment Status</li>
            <li>Property Area</li>
        </ul>
    </div>

    <h2>🖥️ Application Features</h2>

    <div class="box success">
        <ul>
            <li>Sidebar-based user input form</li>
            <li>Clean and modern UI</li>
            <li>Color-coded loan decision output</li>
            <li>Display of individual base model predictions</li>
            <li>Final stacking decision with confidence score</li>
            <li>Business explanation for each prediction</li>
        </ul>
    </div>

    <h2>🚀 How to Run the Project</h2>

    <div class="box">
        <ol>
            <li>Install required libraries:</li>
        </ol>
        <code>pip install streamlit pandas numpy scikit-learn</code>

        <ol start="2">
            <li>Ensure <code>loan.csv</code> is present in the project folder</li>
            <li>Run the application:</li>
        </ol>
        <code>streamlit run app.py</code>
    </div>

    <h2>💼 Business Explanation Logic</h2>

    <div class="box">
        <p>
            The system evaluates loan eligibility by analyzing income level,
            credit history, loan amount, and stability indicators.
            By combining predictions from multiple models, the system reduces
            bias and improves decision reliability.
        </p>

        <p>
            This approach mirrors real-world banking systems, where multiple
            risk assessments are used before approving a loan.
        </p>
    </div>

    <h2>📌 Example Approved Case</h2>

    <div class="box success">
        <ul>
            <li>Credit History: Yes</li>
            <li>Applicant Income: 6000</li>
            <li>Co-applicant Income: 2000</li>
            <li>Loan Amount: 150</li>
            <li>Employment: Salaried</li>
        </ul>
        <p>
            Result: <strong>Loan Approved</strong> with high confidence.
        </p>
    </div>

    <h2>⚠️ Notes</h2>

    <div class="box warning">
        <p>
            Loan approval is heavily influenced by credit history.
            Even high-income applicants may be rejected if credit history is poor.
        </p>
    </div>

    <footer>
        <p>
            © 2026 Smart Loan Approval System | Machine Learning Project
        </p>
    </footer>

</div>

</body>
</html>
