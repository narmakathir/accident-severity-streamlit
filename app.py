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
        'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
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
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Predictions", "Custom Prediction Interface", "Visualizations", "Reports", "User Manual", "Admin Page"])

# --- Home ---
if page == "Home":
    st.title("Predicting Traffic Accident Severity Using Machine Learning")
    st.write("This system analyzes traffic accident data to predict the severity of accidents using various machine learning models.")

# --- Data Analysis ---
elif page == "Data Analysis":
    st.title("Data Analysis")
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Injury Severity Distribution")
    st.bar_chart(df['Injury Severity'].value_counts())

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("### Model Performance")
    performance_df = pd.DataFrame(model_scores, index=["Accuracy", "Precision", "Recall", "F1-Score"]).T
    st.dataframe(performance_df.style.background_gradient(cmap="YlGn"))

# --- Predictions (Generic) ---
elif page == "Predictions":
    st.title("Make Predictions from Dataset")
    selected_model = st.selectbox("Choose Model", list(models.keys()))
    model = models[selected_model]
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(f"### {selected_model} - Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

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

# --- Visualizations Page ---
elif page == "Visualizations":
    st.title("Model Visualizations")

    st.subheader("Confusion Matrix & Classification Report")
    model_name = st.selectbox("Select Model", list(models.keys()))
    selected_model = models[model_name]
    y_pred = selected_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df)

    st.subheader("Feature Importance")
    if hasattr(selected_model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": selected_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(importance_df.set_index("Feature"))
    elif hasattr(selected_model, "coef_"):
        coefs = selected_model.coef_[0]
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": np.abs(coefs)
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(importance_df.set_index("Feature"))
    else:
        st.warning("Feature importance not available for this model.")

    st.subheader("Accident Hotspot Map")
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        st.map(df[['Latitude', 'Longitude']].dropna())
    else:
        st.warning("Latitude and Longitude data not found.")

# --- Reports Page ---
elif page == "Reports":
    st.title("Generated Reports")
    st.write("Only Data Analysis report is available currently.")
    st.write("### Dataset Summary")
    st.dataframe(df.describe())

# --- User Manual ---
elif page == "User Manual":
    st.title("User Manual")
    st.write("""
    **Instructions:**
    - **Data Analysis:** View general statistics, correlation, and model comparison.
    - **Predictions:** Use trained models on test data.
    - **Custom Prediction Interface:** Input your own values to predict accident severity.
    - **Visualizations:** Explore confusion matrix, feature importances, and accident heatmap.
    - **Reports:** Access summary reports.
    - **Admin Page:** Upload new datasets (admin only).
    """)

# --- Admin Page ---
elif page == "Admin Page":
    st.title("Admin Panel")
    uploaded_file = st.file_uploader("Upload new dataset", type=["csv"])
    if uploaded_file:
        st.success("Dataset uploaded. Reload app to use new data.")
