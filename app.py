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

# --- Accident Features ---
accident_related_features = [
    'Driver At Fault', 'Circumstance', 'Driver Distracted By', 'Collision Type',
    'Vehicle Movement', 'Vehicle Going Dir', 'Vehicle First Impact Location',
    'Vehicle Damage Extent', 'Vehicle Body Type', 'Traffic Control',
    'Weather', 'Surface Condition', 'Light', 'Speed Limit', 'Driver Substance Abuse'
]

# --- Load and Cache Dataset ---
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1sVplp_5lFb3AMG5vWRqIltwNazLyM8vH"
    df = pd.read_csv(url)
    return df

@st.cache_data
def preprocess(df):
    drop_cols = ['Report Number', 'Local Case Number', 'Person ID', 'Vehicle ID',
                 'Latitude', 'Longitude', 'Location', 'Driverless Vehicle', 'Parked Vehicle']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    target = 'Injury Severity'
    numeric_cols = df.select_dtypes(include='number').columns.difference([target])
    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])

    X = df.drop(target, axis=1)
    y = df[target]
    return df, X, y

@st.cache_data
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }

    scores = {}
    preds = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds[name] = model
        scores[name] = [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, average='weighted', zero_division=0),
            recall_score(y_test, y_pred, average='weighted', zero_division=0),
            f1_score(y_test, y_pred, average='weighted', zero_division=0)
        ]
    return scores, preds

# --- Load Everything Once ---
df_raw = load_data()
df_cleaned, X, y = preprocess(df_raw)
model_scores, trained_models = train_models(X, y)

# --- Sidebar Navigation ---
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Visualizations"])

# --- Home Page ---
if page == "Home":
    st.title("🚧 Accident Severity Prediction App")
    st.markdown("""
    This application is part of the Final Year Project titled **"Predicting Traffic Accident Severity Using Machine Learning"**.

    ### 📌 Overview
    Traffic accidents are a growing concern due to their impacts on human lives and infrastructure. This system uses **machine learning models** such as Random Forest, XGBoost, Logistic Regression, and ANN to predict accident severity (Minor, Serious, Fatal).  
    
    The predictive model is designed to:
    - Analyze contributory factors (weather, road, vehicle, driver).
    - Assist emergency responders in resource allocation.
    - Identify key risk patterns to improve road safety.

    📚 Dataset Source: [Crash Reporting - Drivers Data](https://catalog.data.gov/dataset/crash-reporting-drivers-data)
    """)

# --- Dataset Page ---
elif page == "Dataset":
    st.title("🗃️ Dataset Overview & Details")
    st.subheader("🔎 Dataset Preview")
    st.dataframe(df_raw.head(), use_container_width=True)

    st.subheader("🧾 Dataset Info")
    st.text(f"Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    missing_vals = df_raw.isnull().sum()
    st.text("Missing Values:")
    st.text(missing_vals[missing_vals > 0])

    st.subheader("📊 Summary Statistics")
    st.dataframe(df_raw.describe(include='all'), use_container_width=True)

    st.subheader("📌 Notes")
    st.markdown("""
    - The dataset includes both categorical and numerical variables.
    - Target column for prediction: **Injury Severity**
    - Common preprocessing steps: missing value imputation, encoding, normalization.
    """)

# --- Visualizations Page ---
elif page == "Visualizations":
    st.title("📈 Visualizations and Feature Importance")

    target_col = 'Injury Severity'
    accident_features = [f for f in accident_related_features if f in df_cleaned.columns]

    # --- Feature Importance ---
    st.subheader("📌 Feature Importance (Accident-Related - Random Forest)")
    rf_model = trained_models['Random Forest']
    importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    accident_imp = importances[accident_features].sort_values(ascending=False)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=accident_imp.values, y=accident_imp.index, ax=ax1)
    ax1.set_title('Accident Feature Importances (Random Forest)')
    st.pyplot(fig1)

    # --- Correlation Heatmap ---
    st.subheader("🔍 Correlation Heatmap")
    corr_matrix = df_cleaned[accident_features + [target_col]].corr()
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
    st.pyplot(fig2)

    # --- Model Comparison Table ---
    st.subheader("📊 Model Comparison Table")
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    score_df = pd.DataFrame(model_scores, index=metrics).T
    st.dataframe(score_df.style.format("{:.2f}"))

    # --- Model Comparison Bar Chart ---
    st.subheader("📉 Model Comparison Chart")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.2
    for i, (name, values) in enumerate(score_df.iterrows()):
        ax3.bar(x + (i - 1.5) * width, values, width, label=name)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.set_ylabel("Score")
    ax3.set_title("Comparison of ML Models")
    ax3.legend()
    st.pyplot(fig3)
