

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

# --- Config ---
st.set_page_config(page_title="Accident Severity Predictor", layout="wide")
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Predictions", "Reports", "User Manual", "Admin"])

# --- Global Config ---
DEFAULT_FILE = "Crash_Reporting.csv"
GDRIVE_URL = "https://drive.google.com/uc?id=1aIzqBWtGg5K20E9xC2FQKfe7r0TSvcV6"

@st.cache_data
def load_default_data():
    if not os.path.exists(DEFAULT_FILE):
        gdown.download(GDRIVE_URL, DEFAULT_FILE, quiet=False)
    return pd.read_csv(DEFAULT_FILE)

def preprocess_data(df, target_col='Injury Severity'):
    drop_cols = ['Report Number', 'Local Case Number', 'Person ID', 'Vehicle ID', 'Latitude', 'Longitude', 'Location', 'Driverless Vehicle', 'Parked Vehicle']
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

# --- Load Data ---
df = load_default_data()

# --- Home Page ---
if page == "Home":
    st.title("🚧 Accident Severity Prediction App")
    st.markdown("""
    This application is part of the Final Year Project titled **"Predicting Traffic Accident Severity Using Machine Learning"**.

    ### 📌 Overview
    This project aims to use **machine learning algorithms** such as Random Forest, XGBoost, Logistic Regression, and ANN
    to classify the severity of traffic accidents (Minor, Serious, or Fatal) using contributory factors.

    📊 Key Features:
    - Data exploration and preprocessing
    - ML-based prediction interface
    - Model evaluation and comparison
    - Feature importance visualization

    📚 Dataset Source: [Crash Reporting - Drivers Data](https://catalog.data.gov/dataset/crash-reporting-drivers-data)
    """)

# --- Data Analysis Page ---
elif page == "Data Analysis":
    st.title("📊 Data Analysis & Model Performance")

    st.subheader("🔍 Data Overview")
    st.dataframe(df.head(), use_container_width=True)

    st.text(f"Dataset Shape: {df.shape}")
    st.text(f"Missing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    st.subheader("📌 Model Training & Evaluation")
    try:
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

        st.dataframe(pd.DataFrame(scores, index=["Accuracy", "Precision", "Recall", "F1-Score"]).T.style.format("{:.2f}"))

        st.subheader("📈 Feature Importance (Random Forest)")
        importances = models['Random Forest'].feature_importances_
        imp_df = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=imp_df.values, y=imp_df.index, ax=ax1)
        ax1.set_title("Feature Importance")
        st.pyplot(fig1)
    except Exception as e:
        st.error(f"Error processing data: {e}")

# --- Predictions Page ---
elif page == "Predictions":
    st.title("📍 Predict Accident Severity")

    input_df = df.drop('Injury Severity', axis=1).head(1).copy()

    st.markdown("Please fill in the accident conditions:")
    user_input = {}
    for col in input_df.columns:
        if df[col].dtype == 'object':
            user_input[col] = st.selectbox(col, df[col].unique())
        else:
            user_input[col] = st.number_input(col, value=float(df[col].median()), format="%.2f")

    input_df = pd.DataFrame([user_input])
    encoded_df = input_df.copy()

    for col in encoded_df.select_dtypes(include='object').columns:
        encoded_df[col] = LabelEncoder().fit(df[col]).transform(encoded_df[col])

    for col in encoded_df.columns:
        if df[col].dtype in ['int64', 'float64']:
            encoded_df[col] = StandardScaler().fit(df[[col]]).transform(encoded_df[[col]])

    model = RandomForestClassifier(random_state=42)
    (X_train, X_test, y_train, y_test), _ = preprocess_data(df)
    model.fit(X_train, y_train)
    pred = model.predict(encoded_df)[0]
    st.success(f"✅ Predicted Severity: **{pred}**")

# --- Reports Page ---
elif page == "Reports":
    st.title("📝 Data Analysis Report")
    st.markdown("""
    ### Summary
    - Dataset used: Crash Reporting Dataset (Gov data)
    - Preprocessing: Imputation, encoding, normalization
    - Models used: RF, XGBoost, LR, ANN
    - Best performance: Generally Random Forest / XGBoost
    - Important Features: Weather, Driver Behavior, Surface Condition, Speed Limit

    📌 Report generation for download is not included in FYP1 scope.
    """)

# --- User Manual Page ---
elif page == "User Manual":
    st.title("📘 User Manual")
    st.markdown("""
    ### Instructions
    - **Home**: Overview and purpose of the app
    - **Data Analysis**: Visual summary + model training
    - **Predictions**: Input features to get predicted severity
    - **Reports**: Summarized results and key observations
    - **Admin Page**: Upload a new dataset (overrides default)

    ⚠️ Default dataset: Crash_Reporting.csv (automatically downloaded)
    """)

# --- Admin Page ---
elif page == "Admin":
    st.title("⚙️ Admin Panel - Upload Dataset")
    uploaded_file = st.file_uploader("📁 Upload new dataset (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.to_csv(DEFAULT_FILE, index=False)
        st.success("✅ Dataset replaced successfully.")
    else:
        st.info("Using default dataset from Google Drive.")
