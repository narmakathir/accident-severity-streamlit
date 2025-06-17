# --- Imports --- 
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# --- Config --- 
st.set_page_config(page_title="Accident Severity Predictor", layout="wide")

# --- Project Overview --- 
PROJECT_OVERVIEW = """
Traffic accidents are a major problem worldwide, causing several fatalities, damage to property, and loss of productivity. Predicting accident severity based on contributors such as weather conditions, road conditions, types of vehicles, and drivers enables the authorities to take necessary actions to minimize the risk and develop better emergency responses. 
 
This project uses machine learning techniques to analyze past traffic data for accident severity prediction and present useful data to improve road safety and management.
"""

# --- Load Dataset --- 
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

    X = df.drop([target_col, 'Location'], axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return df, X, y, X_train, X_test, y_train, y_test, label_encoders

df, X, y, X_train, X_test, y_train, y_test, label_encoders = load_data()

# --- Train Models --- 
@st.cache_resource
def train_models():
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }
    trained_models = {}
    model_scores = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        trained_models[name] = model
        model_scores.append([name, acc*100, prec*100, rec*100, f1*100])

    scores_df = pd.DataFrame(model_scores, columns=['Model', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'])

    return trained_models, scores_df

models, scores_df = train_models()

# --- Side Menu --- 
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Custom Prediction Interface", "Reports", "User Manual", "Admin Page"])

# --- Home --- 
if page == "Home":
    st.title("Traffic Accident Severity Prediction")
    st.write(PROJECT_OVERVIEW)

    st.subheader("Dataset Preview")
    st.dataframe(df.copy().head())    

    st.subheader("Dataset Summary")
    st.write(f"**Number of Records:** {len(df)}")
    st.write(f"**Features:** {list(X.columns)}")

# --- Data Analysis --- 
elif page == "Data Analysis":
    st.title("📊 Data Analysis")
    st.markdown("*Here you can explore key patterns in the data.*")
    st.divider()

    # Injury Distribution
    st.subheader("➥ Injury Severity Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Injury Severity', data=df, ax=ax, palette='coolwarm')
    ax.set_title('Count of Injury Levels')
    st.pyplot(fig)

    st.divider()

    # Hotspot Location
    st.subheader("➥ Hotspot Location")
    if 'Location' in df.columns:
        coords = df['Location'].str.extract(r'\((.*),(.*)\)')
        coords.columns = ['latitude', 'longitude']

        coords = coords.astype(float).dropna()
        if not coords.empty:
            st.map(coords)
        else:
            st.error("No geographic data available.")
    else:
        st.error("Location column not present.")
    st.divider()

    # Correlation Heatmap
    st.subheader("➥ Correlation Heatmap")
    corr = df.select_dtypes(['number']).corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    st.divider()

    # Model Performance (without color fill, just table)
    st.subheader("➥ Model Performance")
    scores_df_percentage = scores_df.copy()
    scores_df_percentage = scores_df_percentage.round(2)
    st.table(scores_df_percentage)

    st.divider()

    # Model Comparison Metrics (Using your code)
    st.subheader("➥ Model Comparison Metrics")
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.2

    fig, ax = plt.subplots()
    ax.bar(x - width*1.5, model_scores['Logistic Regression'], width, label='LR')
    ax.bar(x - width*0.5, model_scores['Random Forest'], width, label='RF')
    ax.bar(x + width*0.5, model_scores['XGBoost'], width, label='XGB')
    ax.bar(x + width*1.5, model_scores['Artificial Neural Network'], width, label='ANN')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.set_title('Model Comparison Metrics')
    ax.set_ylabel('Score (%)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    st.pyplot(fig)

    st.divider()

    # Model-Specific Feature Importances
    st.subheader("➥ Model-Specific Feature Importances")
    model_name = st.selectbox("Select Model", ['Random Forest', 'XGBoost', 'Logistic Regression', 'Artificial Neural Network'], index=0)

    importances = {
        'Random Forest': models['Random Forest'].feature_importances_,
        'XGBoost': models['XGBoost'].feature_importances_,
        'Logistic Regression': np.abs(models['Logistic Regression'].coef_[0]),
        'Artificial Neural Network': np.mean(np.abs(models['Artificial Neural Network'].coefs_[0]), axis=1),
    }
    importances_vals = importances[model_name]
    importances_vals /= importances_vals.sum()

    sorted_idx = np.argsort(importances_vals)[::-1]
    top_features = X.columns[sorted_idx][:10]
    top_vals = importances_vals[sorted_idx][:10]

    fig, ax = plt.subplots()
    sns.barplot(x=top_vals, y=top_features, ax=ax, palette='coolwarm')
    ax.set_title(f'{model_name} Top 10 Features')
    st.pyplot(fig)

# --- Custom Prediction Interface --- 
elif page == "Custom Prediction Interface":
    st.title("Custom Prediction Interface")
    selected_model = st.selectbox("Choose Model for Prediction", list(models.keys()))
    model = models[selected_model]

    input_data = {}
    for col in X.columns:
        if col in label_encoders:
            options = list(label_encoders[col].classes_)
            choice = st.selectbox(f"{col}", options)
            input_data[col] = label_encoders[col].transform([choice])[0]
        else:
            input_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    confidence = np.max(probs) * 100

    severity_label = label_encoders['Injury Severity'].inverse_transform([prediction])[0] if 'Injury Severity' in label_encoders else prediction
    st.success(f"**Predicted Injury Severity:** {severity_label}")
    st.info(f"**Confidence:** {confidence:.2f}%")

# --- Reports --- 
elif page == "Reports":
    st.title("Generated Reports")
    st.write("### Dataset Summary")
    st.dataframe(df.describe())    

# --- User Manual --- 
elif page == "User Manual":
    st.title("User Manual")
    st.write("""
    **Instructions:**
    - **Home:** Project Overview, Dataset preview and Summary.
    - **Data Analysis:** Distribution, Heatmap, Hotspot, Model comparison, and Feature Importances.
    - **Custom Prediction:** Provide custom inputs to predict accident severity.
    - **Reports:** Summary reports of the data.
    - **Admin:** To upload a new dataset (admin only).
    """)

# --- Admin --- 
elif page == "Admin Page":
    st.title("Admin Panel")
    uploaded_file = st.file_uploader("Upload new dataset", type=['csv'])
    if uploaded_file:
        st.success("Dataset uploaded. Reload app to view.")
