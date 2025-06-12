# --- Import Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(page_title="Accident Severity Predictor", layout="wide")
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Predictions", "Reports", "User Manual", "Admin"])

# --- Constants ---
DEFAULT_FILE = "Crash_Reporting.csv"
GDRIVE_URL = "https://drive.google.com/uc?id=1sVplp_5lFb3AMG5vWRqIltwNazLyM8vH"
IMG_DIR = "visualizations"
os.makedirs(IMG_DIR, exist_ok=True)

# --- Load Default Dataset ---
@st.cache_data
def load_default_data():
    if not os.path.exists(DEFAULT_FILE):
        gdown.download(GDRIVE_URL, DEFAULT_FILE, quiet=False)
    return pd.read_csv(DEFAULT_FILE)

# --- Preprocessing Function ---
def preprocess_data(df, target_col='Injury Severity'):
    drop_cols = ['Report Number', 'Local Case Number', 'Person ID', 'Vehicle ID', 
                 'Latitude', 'Longitude', 'Location', 'Driverless Vehicle', 'Parked Vehicle']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    numeric_cols = df.select_dtypes(include='number').columns.difference([target_col])
    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns

# --- Load Dataset ---
df = load_default_data()

# --- Home Page ---
if page == "Home":
    st.title("\U0001F6A7 Accident Severity Prediction App")
    st.markdown("""
    This application is part of the Final Year Project titled **"Predicting Traffic Accident Severity Using Machine Learning"**.

    ### \U0001F4CC Overview
    Predict severity levels (Minor, Serious, Fatal) using ML models like:
    - Random Forest
    - XGBoost
    - Logistic Regression
    - ANN

    📚 Dataset Source: [Crash Reporting - Drivers Data](https://catalog.data.gov/dataset/crash-reporting-drivers-data)
    """)

# --- Data Analysis Page ---
elif page == "Data Analysis":
    st.title("\U0001F4CA Data Analysis & Model Performance")

    st.subheader("\U0001F50D Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("\U0001F4CB Summary")
    st.text(f"Shape: {df.shape}")
    st.text(f"Missing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    (X_train, X_test, y_train, y_test), feature_cols = preprocess_data(df)
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'ANN': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
    }

    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        scores[name] = [
            accuracy_score(y_test, preds),
            precision_score(y_test, preds, average='weighted', zero_division=0),
            recall_score(y_test, preds, average='weighted', zero_division=0),
            f1_score(y_test, preds, average='weighted', zero_division=0)
        ]

    st.subheader("\U0001F4C8 Model Comparison")
    score_df = pd.DataFrame(scores, index=["Accuracy", "Precision", "Recall", "F1-Score"]).T
    st.dataframe(score_df.style.format("{:.2f}"))

    st.subheader("\U0001F50D Feature Importance (Random Forest)")
    rf = models['Random Forest']
    importances = rf.feature_importances_
    imp_df = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

    fig_path = os.path.join(IMG_DIR, "feature_importance.png")
    if not os.path.exists(fig_path):
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=imp_df.values, y=imp_df.index, ax=ax1)
        ax1.set_title("Feature Importance")
        plt.tight_layout()
        fig1.savefig(fig_path)
    st.image(fig_path, caption="Feature Importance")

# --- Predictions Page ---
elif page == "Predictions":
    st.title("\U0001F4CD Predict Accident Severity")

    try:
        sample_input = df.drop(columns=['Injury Severity']).iloc[0].copy()
        user_data = {}
        for col in sample_input.index:
            if df[col].dtype == 'object':
                user_data[col] = st.selectbox(col, sorted(df[col].dropna().astype(str).unique()))
            else:
                user_data[col] = st.number_input(col, value=float(df[col].median()), format="%.2f")

        input_df = pd.DataFrame([user_data])
        for col in input_df.select_dtypes(include='object').columns:
            input_df[col] = LabelEncoder().fit(df[col].astype(str)).transform(input_df[col])

        for col in input_df.select_dtypes(include='number').columns:
            input_df[col] = StandardScaler().fit(df[[col]]).transform(input_df[[col]])

        model = RandomForestClassifier(random_state=42)
        (X_train, X_test, y_train, y_test), _ = preprocess_data(df)
        model.fit(X_train, y_train)
        pred = model.predict(input_df)[0]
        st.success(f"\u2705 Predicted Severity: **{pred}**")
    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")

# --- Reports Page ---
elif page == "Reports":
    st.title("\U0001F4C4 Reports")
    report_text = """
    ## Data Analysis Summary

    - **Models Used**: Random Forest, XGBoost, Logistic Regression, ANN
    - **Features Considered**: Driver behavior, surface condition, weather, etc.
    - **Best Performing Model**: Usually Random Forest or XGBoost
    - **Key Indicators**: Speed Limit, Driver Substance Abuse, Light Condition, etc.
    """
    st.markdown(report_text)

    st.download_button(
        label="📥 Download Report",
        data=report_text,
        file_name="accident_severity_report.txt",
        mime="text/plain"
    )

# --- User Manual Page ---
elif page == "User Manual":
    st.title("\U0001F4D8 User Manual")
    st.markdown("""
    ### How to Use
    - **Home**: Overview of the system
    - **Data Analysis**: Explore and evaluate dataset with ML models
    - **Predictions**: Enter values to get predicted severity
    - **Reports**: Summary of findings and option to download
    - **Admin**: Upload new dataset
    """)

# --- Admin Page ---
elif page == "Admin":
    st.title("⚙️ Admin Panel")
    uploaded = st.file_uploader("📁 Upload New Dataset (CSV)", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        df.to_csv(DEFAULT_FILE, index=False)
        st.success("✅ Dataset replaced. Reload the app to use new data.")
    else:
        st.info("Default dataset is currently in use.")


