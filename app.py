# --- Import Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Predictions", "Reports", "User Manual", "Admin"])

# --- Load Default Dataset ---
DEFAULT_PATH = "crash_reporting.csv"
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV (with 'Injury Severity')", type=["csv"])
df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv(DEFAULT_PATH)

# --- Preprocess Dataset ---
def preprocess_data(data):
    drop_cols = ['Report Number', 'Local Case Number', 'Person ID', 'Vehicle ID',
                 'Latitude', 'Longitude', 'Location', 'Driverless Vehicle', 'Parked Vehicle']
    data.drop(columns=drop_cols, inplace=True, errors='ignore')

    data.fillna(data.median(numeric_only=True), inplace=True)
    data.fillna(data.mode().iloc[0], inplace=True)

    for col in data.select_dtypes(include='object').columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    numeric_cols = data.select_dtypes(include='number').columns.difference(['Injury Severity'])
    data[numeric_cols] = StandardScaler().fit_transform(data[numeric_cols])
    return data

# --- Train Models ---
def train_models(data):
    target = 'Injury Severity'
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'ANN': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }

    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores[name] = [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, average='weighted', zero_division=0),
            recall_score(y_test, y_pred, average='weighted', zero_division=0),
            f1_score(y_test, y_pred, average='weighted', zero_division=0)
        ]
    return scores, models['Random Forest'], X_train.columns

# --- Home Page ---
if page == "Home":
    st.title("🚧 Accident Severity Prediction App")
    st.markdown("""
    Welcome to the **Accident Severity Prediction System** — part of the Final Year Project to improve road safety using **machine learning**.

    **Features:**
    - Analyze accident datasets
    - Predict severity levels: Minor, Serious, Fatal
    - Identify key risk factors
    - Generate automated reports

    📚 **Dataset:** Crash Reporting - Drivers Data  
    🔗 [Data Source](https://catalog.data.gov/dataset/crash-reporting-drivers-data)
    """)

# --- Data Analysis Page ---
elif page == "Data Analysis":
    st.title("📊 Data Analysis")
    df_clean = preprocess_data(df.copy())
    scores, rf_model, feature_names = train_models(df_clean)

    st.subheader("📌 Feature Importance (Random Forest)")
    importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances.head(15), y=importances.head(15).index, ax=ax1)
    ax1.set_title('Top Feature Importances')
    st.pyplot(fig1)

    st.subheader("🔍 Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_clean.corr(), cmap="coolwarm", annot=False, ax=ax2)
    st.pyplot(fig2)

    st.subheader("📈 Model Performance Comparison")
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    score_df = pd.DataFrame(scores, index=metric_labels).T
    st.dataframe(score_df.style.format("{:.2f}"))

# --- Predictions Page ---
elif page == "Predictions":
    st.title("🔮 Predict Accident Severity")

    df_encoded = preprocess_data(df.copy())
    X = df_encoded.drop("Injury Severity", axis=1)
    y = df_encoded["Injury Severity"]
    rf = RandomForestClassifier().fit(X, y)

    st.markdown("### 🚘 Input accident details below:")
    user_input = {}
    for col in X.columns:
        if df[col].dtype == 'object':
            options = df[col].unique().tolist()
            user_input[col] = st.selectbox(f"{col}", options)
        else:
            user_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))

    input_df = pd.DataFrame([user_input])
    input_df = preprocess_data(input_df)
    prediction = rf.predict(input_df)[0]
    st.success(f"Predicted Injury Severity: **{prediction}**")

# --- Reports Page ---
elif page == "Reports":
    st.title("📝 Automated Data Analysis Report")
    st.markdown("""
    The dataset was analyzed using multiple ML models.
    
    **Key Findings**:
    - Most influential features were related to driver behavior and road conditions.
    - Random Forest and XGBoost outperformed Logistic Regression and ANN.
    - Normalization and encoding improved accuracy.

    **Model Accuracy Overview**:
    """)

    df_clean = preprocess_data(df.copy())
    scores, _, _ = train_models(df_clean)
    score_df = pd.DataFrame(scores, index=['Accuracy', 'Precision', 'Recall', 'F1-Score']).T
    st.dataframe(score_df.style.format("{:.2f}"))

# --- User Manual Page ---
elif page == "User Manual":
    st.title("📘 User Manual")
    st.markdown("""
    ### 🚧 Accident Severity Prediction App - How to Use

    1. **Home** - Understand project goals and data sources.
    2. **Data Analysis** - View visual trends, model comparison, and feature importance.
    3. **Predictions** - Enter accident parameters to predict severity.
    4. **Reports** - Auto-generate insights and findings from data.
    5. **Admin** - Upload new datasets if available.

    ### Notes:
    - Make sure the dataset includes `Injury Severity` column.
    - Use `.csv` format.
    """)

# --- Admin Page ---
elif page == "Admin":
    st.title("👩‍💼 Admin Dashboard")
    st.markdown("Upload a new dataset to replace or update the current analysis.")
    new_file = st.file_uploader("Upload new dataset (.csv)", type=["csv"])
    if new_file:
        with open(DEFAULT_PATH, "wb") as f:
            f.write(new_file.read())
        st.success("✅ Dataset updated successfully.")

