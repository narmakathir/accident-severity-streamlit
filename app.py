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
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42)

    return df, X, y, X_train, X_test, y_train, y_test, label_encoders

df, X, y, X_train, X_test, y_train, y_test, label_encoders = load_data()

# --- Train Models --- 
@st.cache_resource
def train_models():
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgboost.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }
    trained_models = {}
    model_scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        trained_models[name] = model
        model_scores[name] = [acc, prec, rec, f1]

    return trained_models, model_scores

models, model_scores = train_models()

# --- Page Navigation --- 
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Custom Prediction Interface", "Reports", "User Manual", "Admin Page"])

# --- Home --- 
if page == "Home":
    st.title("Traffic Accident Severity Prediction")
    st.write("""
    **Project Overview:**  
    Traffic accidents are a major problem causing injuries, fatalities, and damage to property.  
    Machine-learning techniques can help to predict accident severity and enable authorities to respond promptly and efficiently.  
    """)

    st.write("### Dataset Preview")
    st.dataframe(df.copy())    

    st.write("### Dataset Summary")
    st.write(f"- Number of rows: {len(df)}")
    st.write(f"- Number of columns: {len(df.columns)}")
    st.write("- Types of variables:")
    st.write(df.dtypes)

# --- Data Analysis --- 
elif page == "Data Analysis":
    st.title("Data Analysis")
    st.write("### Injury Severity Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Injury Severity', data=df, ax=ax, palette='Blues')
    ax.set_title('Distribution of Injury Severity')
    st.pyplot(fig)

    st.write("### Hotspot Heatmap")
    if 'Location' in df.columns:
        coords = df['Location'].str.extract(r'\((.*),(.*)\)')
        coords.columns = ['Latitude', 'Longitude']

        coords = coords.astype(float).dropna()
        st.map(coords)
    else:
        st.error("Location data not available.")
        
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    corr_matrix = df.select_dtypes(include='number').corr()
    sns.heatmap(corr_matrix, cmap='Blues', annot=False, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    st.write("### Model Performance")
    performance_df = pd.DataFrame(model_scores, index=["Accuracy", "Precision", "Recall", "F1-Score"]).T
    st.dataframe(performance_df.style.background_gradient(cmap='YlGn'))    

    st.write("### Model Comparison")
    fig, ax = plt.subplots()
    performance_df.plot(kind='bar', ax=ax)
    ax.set_title('Model Comparison')
    ax.set_ylabel('Score')
    st.pyplot(fig)

    st.write("### Feature Importances")
    model_name = st.selectbox("Select Model", list(models.keys()))

    if model_name == 'Random Forest':
        importances = models[model_name].feature_importances_
    elif model_name == 'XGBoost':
        importances = models[model_name].feature_importances_
    elif model_name == 'Logistic Regression':
        importances = np.abs(models[model_name].coef_[0]) / np.sum(np.abs(models[model_name].coef_[0]))

    elif model_name == 'Artificial Neural Network':
        importances = np.mean(np.abs(models[model_name].coefs_[0]), axis=1) / np.sum(np.mean(np.abs(models[model_name].coefs_[0]), axis=1))
    else:
        importances = np.ones(len(X.columns))  # fallback

    idx = np.argsort(importances)[::-1]
    top_features = X.columns[idx][:10]
    top_vals = importances[idx][:10]

    fig, ax = plt.subplots()
    sns.barplot(x=top_vals, y=top_features, ax=ax, palette='Blues_d')
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
            input_data[col] = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    confidence = np.max(probs) * 100

    severity_label = label_encoders['Injury Severity'].inverse_transform([prediction])[0] if 'Injury Severity' in label_encoders else prediction
    st.success(f"**Predicted Severity:** {severity_label}")
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
    - **Data Analysis:** View general statistics, distribution, and correlations.
    - **Custom Prediction Interface:** Provide custom inputs to predict accident severity.
    - **Reports:** Access a summary of the data.
    - **Admin Page:** Update or upload a new dataset.
    """)

# --- Admin --- 
elif page == "Admin Page":
    st.title("Admin Panel")
    uploaded_file = st.file_uploader("Upload new dataset", type=["csv"])
    if uploaded_file:
        st.success("Dataset uploaded. Reload app to use new data.")
