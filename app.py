# streamlit_app.py
# ----------------
# Production-ready Streamlit app for EMI_Predict_AI
# - Inference only (NO training)
# - Four tabs: Overview, Eligibility, EMI Estimator, Model Insights
# - Robust numeric cleaning and feature alignment to avoid feature_name errors
# - Dark theme, compact and professional
#
# Save as streamlit_app.py and run with `streamlit run streamlit_app.py`

import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance

# -------------------------
# Configuration
# -------------------------
st.set_page_config(page_title="EMI_Predict_AI", page_icon="💡", layout="wide")
MODELS_DIR = Path("models")
CLASS_PATH = MODELS_DIR / "best_classifier.pkl"
REG_PATH = MODELS_DIR / "best_regressor.pkl"
DATA_PATH = Path("./emi_prediction_dataset.csv")

# -------------------------
# Styling (dark theme)
# -------------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#071026 0%, #061326 100%); color: #E6EEF6; }
    .title { color:#9be7ff; font-weight:800; font-size:34px; text-align:center; }
    .subtitle { color:#cbd5e1; font-size:15px; text-align:center; margin-bottom:18px; }
    .card { background: rgba(255,255,255,0.03); border-radius:12px; padding:14px; margin-bottom:12px; }
    .metric-big { font-size:20px; font-weight:700; color:#a7f3d0; }
    .metric-sub { color:#e2e8f0; }
    .btn-primary > button { background: linear-gradient(90deg,#7e22ce,#06b6d4); color:white; border:none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Utility: safe numeric conversion
# -------------------------
def to_numeric_safe(series):
    """Clean a pandas Series that may contain strings like '₹50,000' or '50,000' and convert to numeric."""
    s = series.astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False)
    s = s.str.replace("-", "0", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

# -------------------------
# Load models (inference-only)
# -------------------------
@st.cache_resource
def load_models():
    clf = None
    reg = None
    clf_err = None
    reg_err = None
    try:
        if CLASS_PATH.exists():
            with open(CLASS_PATH, "rb") as f:
                clf = pickle.load(f)
    except Exception as e:
        clf_err = e
    try:
        if REG_PATH.exists():
            with open(REG_PATH, "rb") as f:
                reg = pickle.load(f)
    except Exception as e:
        reg_err = e
    return clf, reg, clf_err, reg_err

clf, reg, clf_err, reg_err = load_models()

# -------------------------
# Load dataset (optional)
# -------------------------
@st.cache_data
def load_dataset(path: Path):
    if path.exists():
        try:
            df = pd.read_csv(path, low_memory=False)
            return df
        except Exception:
            # try reading with utf-8-sig
            try:
                df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
                return df
            except Exception:
                return None
    return None

raw_df = load_dataset(DATA_PATH)

# -------------------------
# Feature engineering (inference-time)
# -------------------------
def create_engineered_features(df_input):
    """Given df_input (pandas DataFrame), return a copy with engineered numeric features.
    Handles string numeric cleaning and fills missing columns with sensible defaults.
    """
    df = df_input.copy()

    # Ensure required numeric columns exist and are numeric
    numeric_cols = [
        "monthly_salary",
        "current_emi_amount",
        "requested_amount",
        "requested_tenure",
        "credit_score",
        "monthly_rent",
        "groceries_utilities",
        "travel_expenses",
        "school_fees",
    ]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0.0
        # Clean and coerce
        df[c] = to_numeric_safe(df[c])

    # Ensure other possibly used numeric columns exist
    optional_numeric = [
        "college_fees",
        "other_monthly_expenses",
        "existing_loans",
        "bank_balance",
        "emergency_fund",
        "age",
        "years_of_employment",
        "dependents",
        "family_size",
    ]
    for c in optional_numeric:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = to_numeric_safe(df[c])

    # Derived features
    df["total_expenses"] = df[
        ["groceries_utilities", "travel_expenses", "school_fees", "monthly_rent", "other_monthly_expenses"]
    ].sum(axis=1)

    eps = 1e-9
    # Avoid division by zero by replacing 0 with small eps for denominators
    monthly_salary_safe = df["monthly_salary"].replace(0, eps)

    df["dti_ratio"] = (df["current_emi_amount"] + df["requested_amount"]) / monthly_salary_safe
    df["savings_rate"] = (df["monthly_salary"] - df["total_expenses"]) / monthly_salary_safe
    df["expense_to_income"] = df["total_expenses"] / monthly_salary_safe
    df["credit_health"] = df["credit_score"] / 1000.0

    # For any remaining NaNs (edge cases), fill with 0
    df = df.fillna(0.0)
    return df

# ----------------------------------------------
# Encode categorical features before prediction
# ----------------------------------------------
def encode_categorical_features(df):
    """Convert categorical string features to numeric codes to match model training."""
    # Define simple mappings (must match those used during training)
    mappings = {
        "gender": {"Male": 0, "Female": 1},
        "marital_status": {"Single": 0, "Married": 1},
        "education": {"HighSchool": 0, "Bachelors": 1, "Masters": 2},
        "employment_type": {"Salaried": 0, "Self-Employed": 1},
        "company_type": {"Private": 0, "Public": 1, "Government": 2},
        "house_type": {"Owned": 0, "Rented": 1},
        "emi_scenario": {"Personal": 0, "Business": 1, "Home": 2}
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    return df

# -------------------------
# Align features with model to avoid feature_names mismatch
# -------------------------
def align_features_with_model(model, df_input):
    df_temp = df_input.copy()
    # If model exposes feature_names_in_ (sklearn >= 1.0), use it
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        for c in expected:
            if c not in df_temp.columns:
                # set default neutral values for missing columns
                df_temp[c] = 0.0
        # Return columns in the exact order
        return df_temp[expected]
    # If not available, just return numeric columns in df_input
    return df_temp.select_dtypes(include=[np.number])

# -------------------------
# Helpers for predictions and label display
# -------------------------
def format_label(model, raw_pred):
    """Try to convert raw_pred to human-readable label using model.classes_ if present."""
    try:
        if hasattr(model, "classes_"):
            classes = model.classes_
            # If classes are strings, return directly
            if isinstance(classes[0], (str,)):
                return classes[int(raw_pred)]
            else:
                # If classes are numeric and raw_pred is numeric index -> attempt map
                # But raw_pred may be already the class value
                if raw_pred in classes:
                    return raw_pred
                else:
                    # try to index
                    try:
                        return classes[int(raw_pred)]
                    except Exception:
                        return raw_pred
        return raw_pred
    except Exception:
        return raw_pred

# -------------------------
# UI: Tabs (4 pages)
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "🔍 Eligibility", "💰 EMI Estimator", "📊 Model Insights"])

# -------------------------
# Tab 1: Overview
# -------------------------
with tab1:
    st.markdown("<div class='title'>EMI_Predict_AI — Dashboard Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Welcome — dataset summary and quick metrics</div>", unsafe_allow_html=True)

    if raw_df is None:
        st.info("No dataset found at './emi_prediction_dataset.csv'. To show dataset metrics, place your CSV at this path.")
    else:
        display_df = raw_df.head(10)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='metric-big'>Total records</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-sub'>{display_df.shape[0]:,} (preview)</div>", unsafe_allow_html=True)
        with c2:
            if "monthly_salary" in raw_df.columns:
                mean_sal = int(to_numeric_safe(raw_df["monthly_salary"]).mean())
            else:
                mean_sal = 0
            st.markdown("<div class='metric-big'>Avg Monthly Salary</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-sub'>₹ {mean_sal:,}</div>", unsafe_allow_html=True)
        with c3:
            if "credit_score" in raw_df.columns:
                mean_cs = int(to_numeric_safe(raw_df["credit_score"]).mean())
            else:
                mean_cs = 0
            st.markdown("<div class='metric-big'>Avg Credit Score</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-sub'>{mean_cs}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Data sample")
        st.dataframe(display_df, use_container_width=True)

        # class distribution if present
        if "emi_eligibility" in raw_df.columns:
            st.subheader("EMI Eligibility distribution (sample)")
            try:
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.countplot(x="emi_eligibility", data=raw_df.sample(min(2000, len(raw_df))), palette="viridis", ax=ax)
                ax.set_title("EMI Eligibility Distribution")
                st.pyplot(fig)
            except Exception:
                st.info("Unable to plot distribution (check dataset format).")

# -------------------------
# Tab 2: Eligibility Predictor
# -------------------------
with tab2:
    st.markdown("<div class='title'>Eligibility Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Enter applicant details for a prediction</div>", unsafe_allow_html=True)

    if clf is None:
        st.warning("Classification model not found in ./models/. Place best_classifier.pkl and refresh app.")
        if clf_err:
            st.caption(f"Classifier load error: {clf_err}")

    with st.form("eligibility_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            education = st.selectbox("Education", ["HighSchool", "Bachelors", "Masters"])
        with c2:
            monthly_salary = st.number_input("Monthly Salary (₹)", min_value=0.0, value=50000.0, step=1000.0)
            employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
            years_of_employment = st.number_input("Years of Employment", min_value=0, max_value=60, value=3)
            company_type = st.selectbox("Company Type", ["Private", "Public", "Government"])
        with c3:
            house_type = st.selectbox("House Type", ["Owned", "Rented"])
            dependents = st.number_input("Dependents", min_value=0, max_value=20, value=0)
            family_size = st.number_input("Family Size", min_value=1, max_value=20, value=3)
            existing_loans = st.number_input("Existing loans (count)", min_value=0, value=0)

        current_emi_amount = st.number_input("Current EMI Amount (₹)", min_value=0.0, value=5000.0, step=500.0)
        requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=1000.0, value=100000.0, step=1000.0)
        requested_tenure = st.number_input("Requested Tenure (months)", min_value=6, value=24, step=1)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0.0, value=8000.0, step=500.0)
        groceries = st.number_input("Groceries & Utilities (₹)", min_value=0.0, value=4000.0, step=100.0)
        travel = st.number_input("Travel Expenses (₹)", min_value=0.0, value=1500.0, step=100.0)
        school_fees = st.number_input("School Fees (₹)", min_value=0.0, value=0.0, step=100.0)

        submitted_elig = st.form_submit_button("Predict Eligibility")

    if submitted_elig:
        row = {
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "company_type": company_type,
            "house_type": house_type,
            "monthly_rent": monthly_rent,
            "family_size": family_size,
            "dependents": dependents,
            "school_fees": school_fees,
            "college_fees": 0,
            "travel_expenses": travel,
            "groceries_utilities": groceries,
            "other_monthly_expenses": 0,
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "credit_score": credit_score,
            "bank_balance": 0,
            "emergency_fund": 0,
            "emi_scenario": "Personal",
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
        }
        df_input = pd.DataFrame([row])
        df_input = create_engineered_features(df_input)
        df_input = encode_categorical_features(df_input)

        if clf is None:
            st.error("No classifier available to predict. Train the model via notebook and save to ./models/.")
        else:
            try:
                X_in = align_features_with_model(clf, df_input)
                pred_raw = clf.predict(X_in)[0]
                pred_label = format_label(clf, pred_raw)
                prob = None
                if hasattr(clf, "predict_proba"):
                    try:
                        prob = clf.predict_proba(X_in)[0]
                    except Exception:
                        prob = None

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-big'>Eligibility: {pred_label}</div>", unsafe_allow_html=True)
                if prob is not None:
                    st.write("Class probabilities:", {str(k): float(v) for k, v in zip(getattr(clf, "classes_", []), prob)})
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -------------------------
# Tab 3: EMI Estimator
# -------------------------
with tab3:
    st.markdown("<div class='title'>EMI Affordability Estimator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Predict maximum affordable EMI (monthly)</div>", unsafe_allow_html=True)

    if reg is None:
        st.warning("Regression model not found in ./models/. Place best_regressor.pkl and refresh.")
        if reg_err:
            st.caption(f"Regressor load error: {reg_err}")

    with st.form("emi_form", clear_on_submit=False):
        s1, s2 = st.columns(2)
        with s1:
            monthly_salary_r = st.number_input("Monthly Salary (₹)", min_value=0.0, value=50000.0, step=1000.0)
            current_emi_amount_r = st.number_input("Current EMI Amount (₹)", min_value=0.0, value=5000.0, step=500.0)
            requested_amount_r = st.number_input("Requested Loan Amount (₹)", min_value=1000.0, value=100000.0, step=1000.0)
        with s2:
            requested_tenure_r = st.number_input("Requested Tenure (months)", min_value=6, value=24, step=1)
            credit_score_r = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
            monthly_rent_r = st.number_input("Monthly Rent (₹)", min_value=0.0, value=8000.0, step=500.0)

        groceries_r = st.number_input("Groceries & Utilities (₹)", min_value=0.0, value=4000.0, step=100.0)
        travel_r = st.number_input("Travel Expenses (₹)", min_value=0.0, value=1500.0, step=100.0)
        school_fees_r = st.number_input("School Fees (₹)", min_value=0.0, value=0.0, step=100.0)
        submitted_reg = st.form_submit_button("Estimate Max EMI")

    if submitted_reg:
        df_input_r = pd.DataFrame(
            [
                {
                    "monthly_salary": monthly_salary_r,
                    "current_emi_amount": current_emi_amount_r,
                    "requested_amount": requested_amount_r,
                    "requested_tenure": requested_tenure_r,
                    "credit_score": credit_score_r,
                    "monthly_rent": monthly_rent_r,
                    "groceries_utilities": groceries_r,
                    "travel_expenses": travel_r,
                    "school_fees": school_fees_r,
                }
            ]
        )
        df_input_r = create_engineered_features(df_input_r)
        df_input_r = encode_categorical_features(df_input_r)

        if reg is None:
            st.error("No regressor available to predict. Train and save model to ./models/.")
        else:
            try:
                X_in_r = align_features_with_model(reg, df_input_r)
                pred_emi = reg.predict(X_in_r)[0]
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-big'>Predicted Max Monthly EMI: ₹ {pred_emi:,.0f}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                pct = (pred_emi / (monthly_salary_r + 1)) * 100
                st.write(f"Predicted EMI is {pct:.1f}% of monthly salary.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -------------------------
# Tab 4: Model Insights
# -------------------------
with tab4:
    st.markdown("<div class='title'>Model Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Evaluation charts & feature importance (read-only)</div>", unsafe_allow_html=True)

    if raw_df is None:
        st.info("No dataset available for model insight sampling. Place emi_prediction_dataset.csv in app folder.")
    else:
        # create engineered copy for evaluation/sampling
        sample = raw_df.sample(n=min(2000, len(raw_df)), random_state=42).copy()
        sample = create_engineered_features(sample)

        if clf is not None and "emi_eligibility" in raw_df.columns:
            st.subheader("Classification: Confusion Matrix (sample)")
            try:
                Xs = align_features_with_model(clf, sample)
                y_true = sample["emi_eligibility"]
                # If classifier was trained with encoded labels, clf.predict will return encoded or original labels accordingly.
                y_pred = clf.predict(Xs)
                cm = confusion_matrix(y_true, y_pred, labels=getattr(clf, "classes_", np.unique(y_true)))
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title("Confusion Matrix (sample)")
                st.pyplot(fig)
            except Exception as e:
                st.info(f"Unable to compute confusion matrix: {e}")

            # ROC curves if probabilities available
            if hasattr(clf, "predict_proba"):
                try:
                    y_score = clf.predict_proba(Xs)
                    from sklearn.preprocessing import label_binarize

                    classes = clf.classes_
                    y_true_b = label_binarize(y_true, classes=classes)
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    for i, cls in enumerate(classes):
                        fpr, tpr, _ = roc_curve(y_true_b[:, i], y_score[:, i])
                        ax2.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr, tpr):.2f})")
                    ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
                    ax2.set_xlabel("False Positive Rate")
                    ax2.set_ylabel("True Positive Rate")
                    ax2.set_title("ROC Curves (sample)")
                    ax2.legend()
                    st.pyplot(fig2)
                except Exception as e:
                    st.info(f"ROC plot skipped: {e}")
        else:
            st.info("Classifier or target labels missing — cannot show classification insights.")

        # Feature importance via permutation (model-agnostic)
        st.subheader("Feature importance (permutation importance)")
        chosen_model = None
        if reg is not None:
            chosen_model = reg
            st.write("Using regressor for importance (fast sample).")
        elif clf is not None:
            chosen_model = clf
            st.write("Using classifier for importance (fast sample).")

        if chosen_model is not None:
            try:
                # Build numeric X and y for permutation importance
                X_perm = sample.select_dtypes(include=[np.number]).fillna(0)
                # Align features with model
                X_perm = align_features_with_model(chosen_model, X_perm)
                # Define y
                if chosen_model is reg and "max_monthly_emi" in sample.columns:
                    y_perm = sample["max_monthly_emi"]
                elif "emi_eligibility" in raw_df.columns:
                    y_perm = sample["emi_eligibility"]
                else:
                    y_perm = None

                if y_perm is None:
                    st.info("No target column available to compute permutation importance.")
                else:
                    perm = permutation_importance(chosen_model, X_perm, y_perm, n_repeats=6, random_state=42, n_jobs=1)
                    importances = perm.importances_mean
                    idx = np.argsort(importances)[::-1][:15]
                    top_feats = X_perm.columns[idx]
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(x=importances[idx], y=top_feats, ax=ax)
                    ax.set_title("Top permutation importances")
                    st.pyplot(fig)
            except Exception as e:
                st.info(f"Permutation importance unavailable: {e}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("EMI_Predict_AI • Deployment-only app. Place models in ./models/ then refresh.")
