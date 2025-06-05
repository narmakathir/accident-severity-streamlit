# --- Import Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(page_title="Accident Severity Predictor", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Visualizations"])

# --- Upload CSV File ---
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV (with 'Injury Severity')", type=["csv"])

# --- Accident Features ---
accident_related_features = [
    'Driver At Fault', 'Circumstance', 'Driver Distracted By', 'Collision Type',
    'Vehicle Movement', 'Vehicle Going Dir', 'Vehicle First Impact Location',
    'Vehicle Damage Extent', 'Vehicle Body Type', 'Traffic Control',
    'Weather', 'Surface Condition', 'Light', 'Speed Limit', 'Driver Substance Abuse'
]

# --- Home Page ---
if page == "Home":
    st.title("🚧 Accident Severity Prediction App")
    st.markdown("""
    This application is part of the Final Year Project titled **"Predicting Traffic Accident Severity Using Machine Learning "**.

    ### 📌 Overview
    Traffic accidents are a growing concern due to their impacts on human lives and infrastructure. This system uses **machine learning models** such as Random Forest, XGBoost, Logistic Regression, and ANN to predict accident severity (e.g., Minor, Serious, or Fatal).  
    
    The predictive model is designed to:
    - Analyze contributory factors (weather, road, vehicle, driver).
    - Assist emergency responders in resource allocation.
    - Identify key risk patterns to improve road safety.

    📚 Dataset Source: [Crash Reporting - Drivers Data](https://catalog.data.gov/dataset/crash-reporting-drivers-data)
    """)

# --- Dataset Page ---
elif page == "Dataset":
    st.title("🗃️ Dataset Overview & Details")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("🔎 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("🧾 Dataset Info")
        st.text(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        st.text("Missing Values:")
        st.text(df.isnull().sum()[df.isnull().sum() > 0])

        st.subheader("📊 Summary Statistics")
        st.dataframe(df.describe(include='all'), use_container_width=True)

        st.subheader("📌 Notes")
        st.markdown("""
        - The dataset includes both categorical and numerical variables.
        - Target column for prediction: **Injury Severity**
        - Common preprocessing steps: missing value imputation, encoding, normalization.
        """)
    else:
        st.info("📁 Please upload the dataset to view its content.")

# --- Visualizations Page ---
elif page == "Visualizations":
    st.title("📈 Visualizations and Feature Importance")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Drop irrelevant columns
        drop_cols = ['Report Number', 'Local Case Number', 'Person ID', 'Vehicle ID',
                     'Latitude', 'Longitude', 'Location', 'Driverless Vehicle', 'Parked Vehicle']
        df.drop(columns=drop_cols, inplace=True, errors='ignore')

        # Handle missing values
        df.fillna(df.median(numeric_only=True), inplace=True)
        df.fillna(df.mode().iloc[0], inplace=True)

        # Encode categoricals
        for col in df.select_dtypes(include='object').columns:
            df[col] = LabelEncoder().fit_transform(df[col])

        # Normalize
        target_col = 'Injury Severity'
        if target_col not in df.columns:
            st.error("❌ 'Injury Severity' column not found.")
            st.stop()

        numeric_cols = df.select_dtypes(include='number').columns.difference([target_col])
        df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])

        # Train/Test split
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- Train and Score All Models ---
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
        }

        model_scores = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_scores[name] = [
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, average='weighted', zero_division=0),
                recall_score(y_test, y_pred, average='weighted', zero_division=0),
                f1_score(y_test, y_pred, average='weighted', zero_division=0)
            ]

        # Feature Importance - Random Forest
        st.subheader("📌 Feature Importance (Accident-Related - Random Forest)")
        all_importances = pd.Series(models['Random Forest'].feature_importances_, index=X.columns)
        accident_features = [f for f in accident_related_features if f in df.columns]
        imp_filtered = all_importances[accident_features].sort_values(ascending=False)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=imp_filtered, y=imp_filtered.index, ax=ax1)
        ax1.set_title('Accident Feature Importances (Random Forest)')
        st.pyplot(fig1)

        # Correlation Heatmap
        st.subheader("🔍 Correlation Heatmap")
        corr_matrix = df[accident_features + [target_col]].corr()
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
        st.pyplot(fig2)

        # Display Scores
        st.subheader("📊 Model Comparison")
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        scores = pd.DataFrame(model_scores, index=metrics).T
        st.dataframe(scores.style.format("{:.2f}"))

        # Bar Chart for Model Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(metrics))
        width = 0.2
        ax.bar(x - 1.5*width, scores.iloc[0], width, label='RF')
        ax.bar(x - 0.5*width, scores.iloc[1], width, label='XGB')
        ax.bar(x + 0.5*width, scores.iloc[2], width, label='LR')
        ax.bar(x + 1.5*width, scores.iloc[3], width, label='ANN')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        st.pyplot(fig)

    else:
        st.info("📁 Please upload the dataset to generate visualizations.")
