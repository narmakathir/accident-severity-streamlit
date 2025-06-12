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
    st.title("🚧 Accident Severity Prediction App")
    st.markdown("""
    This application is part of the Final Year Project titled **"Predicting Traffic Accident Severity Using Machine Learning"**.

    ### 📌 Overview
    This system predicts the severity of traffic accidents using machine learning algorithms such as:
    - Random Forest
    - XGBoost
    - Logistic Regression
    - Artificial Neural Network (ANN)

    📊 Based on:
    - Road conditions
    - Driver behavior
    - Weather
    - Vehicle factors

    📚 Dataset Source: [Crash Reporting - Drivers Data](https://catalog.data.gov/dataset/crash-reporting-drivers-data)
    """)

# --- Data Analysis Page ---
elif page == "Data Analysis":
    st.title("📊 Data Analysis & Model Performance")

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📋 Summary")
    st.text(f"Shape: {df.shape}")
    st.text(f"Missing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

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

        st.subheader("📈 Model Comparison")
        score_df = pd.DataFrame(scores, index=["Accuracy", "Precision", "Recall", "F1-Score"]).T
        st.dataframe(score_df.style.format("{:.2f}"))

        st.subheader("🔎 Feature Importance (Random Forest)")
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
    st.markdown("Fill in the details below to predict the severity of an accident:")

    sample_input = df.drop(columns=['Injury Severity']).iloc[0].copy()
    user_data = {}
    for col in sample_input.index:
        if df[col].dtype == 'object':
            user_data[col] = st.selectbox(col, sorted(df[col].dropna().unique()))
        else:
            user_data[col] = st.number_input(col, value=float(df[col].median()), format="%.2f")

    input_df = pd.DataFrame([user_data])
    encoded_df = input_df.copy()
    for col in encoded_df.select_dtypes(include='object').columns:
        encoded_df[col] = LabelEncoder().fit(df[col]).transform(encoded_df[col])

    for col in encoded_df.select_dtypes(include='number').columns:
        encoded_df[col] = StandardScaler().fit(df[[col]]).transform(encoded_df[[col]])

    model = RandomForestClassifier(random_state=42)
    (X_train, X_test, y_train, y_test), _ = preprocess_data(df)
    model.fit(X_train, y_train)
    pred = model.predict(encoded_df)[0]
    st.success(f"✅ Predicted Severity: **{pred}**")

# --- Reports Page ---
elif page == "Reports":
    st.title("📄 Reports")
    st.markdown("""
    ### 📊 Data Analysis Summary

    - **Models Used**: Random Forest, XGBoost, Logistic Regression, ANN
    - **Features Considered**: Driver behavior, surface condition, weather, vehicle type, etc.
    - **Best Performing Model**: Usually Random Forest or XGBoost
    - **Key Risk Indicators**: High speed, adverse weather, certain vehicle types

    📌 *Note: Report download/export will be available in FYP2 if needed.*
    """)

# --- User Manual Page ---
elif page == "User Manual":
    st.title("📘 User Manual")
    st.markdown("""
    ### App Navigation Guide

    - **Home**: Overview of project and goals.
    - **Data Analysis**: View dataset, train models, see performance and feature importance.
    - **Predictions**: Input real-world accident scenario to get severity prediction.
    - **Reports**: Summary of analysis and key findings.
    - **Admin**: Upload your own dataset for analysis.
    
    📌 *Ensure your dataset includes an 'Injury Severity' column.*
    """)

# --- Admin Page ---
elif page == "Admin":
    st.title("⚙️ Admin Panel")
    uploaded = st.file_uploader("📁 Upload New Dataset (CSV)", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        df.to_csv(DEFAULT_FILE, index=False)
        st.success("✅ Dataset replaced. Reload the app to see changes.")
    else:
        st.info("Default dataset in use from Google Drive.")

