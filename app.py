# === Import Libraries ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# === Streamlit Page Config ===
st.set_page_config(page_title="Accident Severity Prediction App", layout="wide")

# === Load & Preprocess Dataset ===
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/narmakathir/accident-severity-streamlit/main/filtered_crash_data.csv'
    df = pd.read_csv(url)

    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col != 'Location':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    target_col = 'Injury Severity'
    numeric_cols = df.select_dtypes(include='number').columns.difference([target_col])
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, label_encoders

df, label_encoders = load_data()
target_col = 'Injury Severity'

# === Sidebar Navigation ===
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Visualizations"])

# === Home Page ===
if page == "Home":
    st.title("🚧 Accident Severity Prediction App")
    st.markdown("""
    This application is part of the Final Year Project titled **"Predicting Traffic Accident Severity Using Machine Learning"**.

    ### 📌 Overview
    Traffic accidents are a growing concern due to their impacts on human lives and infrastructure. This system uses **machine learning models** such as Random Forest, XGBoost, Logistic Regression, and ANN to predict accident severity (e.g., Minor, Serious, or Fatal).  
    
    The predictive model is designed to:
    - Analyze contributory factors (weather, road, vehicle, driver).
    - Assist emergency responders in resource allocation.
    - Identify key risk patterns to improve road safety.

    📚 Dataset Source: [GitHub - NarmaKathir](https://github.com/narmakathir/accident-severity-streamlit)
    """)

# === Dataset Page ===
elif page == "Dataset":
    st.title("🗃️ Dataset Overview & Details")

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
    - Columns like `Location` were retained for hotspot plotting.
    - Target column for prediction: **Injury Severity**
    - Preprocessing includes: duplicate removal, missing value imputation, encoding, normalization.
    """)

# === Visualizations Page ===
elif page == "Visualizations":
    st.title("📈 Visualizations, Model Performance & Insights")

    # Features to visualize
    eda_cols = [
        'Driver At Fault', 'Driver Distracted By', 'Vehicle Damage Extent',
        'Traffic Control', 'Weather', 'Surface Condition', 'Light',
        'Speed Limit', 'Driver Substance Abuse'
    ]
    eda_in_df = [col for col in eda_cols if col in df.columns]

    # Train-test split
    X = df.drop([target_col, 'Location'], axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Model Training ===
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        "Artificial Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=42)
    }

    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_scores[name] = [
            accuracy_score(y_test, y_pred)*100,
            precision_score(y_test, y_pred, average='weighted')*100,
            recall_score(y_test, y_pred, average='weighted')*100,
            f1_score(y_test, y_pred, average='weighted')*100
        ]

    # === Distribution Plot ===
    st.subheader("🔹 Distribution of Injury Severity")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x=target_col, ax=ax1)
    ax1.set_title("Distribution of Injury Severity")
    st.pyplot(fig1)

    # === Heatmap ===
    st.subheader("🔹 Correlation Heatmap (Accident Features)")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[eda_in_df + [target_col]].corr(), cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    # === Hotspot Plot ===
    st.subheader("🔹 Accident Hotspot Map (Density)")
    df[['Latitude', 'Longitude']] = df['Location'].str.extract(r'\(([^,]+),\s*([^)]+)\)').astype(float)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.kdeplot(x=df['Longitude'], y=df['Latitude'], cmap='Reds', fill=True, alpha=0.6, ax=ax3)
    ax3.set_title("Accident Hotspot Heatmap")
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    st.pyplot(fig3)

    # === Model Comparison Table ===
    st.subheader("🔹 Model Performance Table")
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    scores_df = pd.DataFrame(model_scores, index=metrics).T
    st.dataframe(scores_df.style.format("{:.2f}"))

    # === Model Comparison Chart ===
    st.subheader("🔹 Model Comparison Bar Chart")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.2
    for i, (model_name, scores) in enumerate(model_scores.items()):
        ax4.bar(x + width*i - 1.5*width, scores, width, label=model_name[:3])
    ax4.set_title("Model Performance Comparison")
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.set_ylabel("Score (%)")
    ax4.legend()
    st.pyplot(fig4)

    # === Feature Importance ===
    st.subheader("🔹 Feature Importances (Random Forest)")
    rf_model = models["Random Forest"]
    importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    filtered = importances[eda_in_df].sort_values(ascending=False)
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=filtered.values, y=filtered.index, ax=ax5)
    ax5.set_title("Top Accident-Related Feature Importances (RF)")
    st.pyplot(fig5)
