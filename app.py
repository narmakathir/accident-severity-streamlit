# Full updated Streamlit app code with:
# 1. Working geographic mapping using folium.
# 2. Dark-themed plots.
# 3. Admin page for uploading new dataset and reloading.

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import warnings
import io
import tempfile

warnings.filterwarnings('ignore')

# --- Config ---
st.set_page_config(page_title="Accident Severity Predictor", layout="wide")
sns.set_style("darkgrid")
PALETTE = sns.color_palette("viridis", as_cmap=False)

@st.cache_data(persist="disk")
def normalize_and_prepare(df):
    def normalize_categories(df):
        mappings = {
            'Weather Condition': {'Raining': 'Rain', 'Rainy': 'Rain', 'Drizzling': 'Rain', 'Sun': 'Sunny', 'Clear': 'Sunny', 'Foggy': 'Fog', 'Overcast': 'Cloudy'},
            'Road Condition': {'Wet': 'Wet', 'Dry': 'Dry', 'Snowy': 'Snow/Ice', 'Snow/Ice': 'Snow/Ice', 'Icy': 'Snow/Ice'},
            'Light Condition': {'Dark - No Street Lights': 'Dark', 'Dark - Street Lights Off': 'Dark', 'Dark - Street Lights On': 'Dark', 'Daylight': 'Daylight'}
        }
        for col, replacements in mappings.items():
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title().replace(replacements)
        return df

    df = df.copy()
    df['Location_Original'] = df['Location']
    df = normalize_categories(df)

    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.title()
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

@st.cache_data(persist="disk")
def load_data(uploaded_file=None):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        url = 'https://raw.githubusercontent.com/narmakathir/accident-severity-streamlit/main/filtered_crash_data.csv'
        df = pd.read_csv(url)

    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    return normalize_and_prepare(df)

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

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Prediction", "Reports", "Admin Upload", "Help"])

# Load data
if 'df' not in st.session_state:
    df, X, y, X_train, X_test, y_train, y_test, label_encoders = load_data()
    models, scores_df = train_models()
    st.session_state.update(locals())

# Home
if page == "Home":
    st.title("Traffic Accident Severity Prediction")
    st.write("""
        Traffic accidents cause significant fatalities, damage, and loss. This tool predicts accident severity using machine learning on historical traffic data.
    """)
    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.df.head())

# Data Analysis
elif page == "Data Analysis":
    st.title("Data Analysis & Insights")
    st.subheader("Injury Severity Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Injury Severity', data=st.session_state.df, palette=PALETTE, ax=ax)
    st.pyplot(fig)

    st.subheader("Hotspot Location")
    coords = st.session_state.df['Location_Original'].astype(str).str.extract(r'\(([-\d.]+),\s*([-\d.]+)\)')
    coords.columns = ['latitude', 'longitude']
    coords = coords.astype(float).dropna()
    if not coords.empty:
        sample_coords = coords.sample(min(1000, len(coords)), random_state=42)
        folium_map = folium.Map(location=[sample_coords['latitude'].mean(), sample_coords['longitude'].mean()], zoom_start=11, tiles='CartoDB dark_matter')
        for _, row in sample_coords.iterrows():
            folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=3, color='red', fill=True).add_to(folium_map)
        st_folium(folium_map, width=700, height=500)
    else:
        st.warning("No valid coordinates found in Location.")

    st.subheader("Correlation Heatmap")
    corr = st.session_state.df.select_dtypes(include='number').corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, cmap='viridis', ax=ax)
    st.pyplot(fig)

    st.subheader("Model Performance")
    st.table(st.session_state.scores_df.round(2))

    st.subheader("Model Comparison")
    performance_df = st.session_state.scores_df.set_index('Model')
    fig, ax = plt.subplots()
    performance_df.plot(kind='bar', ax=ax, color=PALETTE.as_hex())
    st.pyplot(fig)

    st.subheader("Top Features by Model")
    model_name = st.selectbox("Choose model", list(st.session_state.models.keys()))
    importances = {
        'Random Forest': st.session_state.models['Random Forest'].feature_importances_,
        'XGBoost': st.session_state.models['XGBoost'].feature_importances_,
        'Logistic Regression': np.abs(st.session_state.models['Logistic Regression'].coef_[0]),
        'Artificial Neural Network': np.mean(np.abs(st.session_state.models['Artificial Neural Network'].coefs_[0]), axis=1),
    }
    top_idx = np.argsort(importances[model_name])[::-1][:10]
    fig, ax = plt.subplots()
    sns.barplot(x=importances[model_name][top_idx], y=st.session_state.X.columns[top_idx], ax=ax, palette=PALETTE)
    st.pyplot(fig)

# Prediction
elif page == "Prediction":
    st.title("Custom Prediction")
    selected_model = st.selectbox("Choose Model", list(st.session_state.models.keys()))
    model = st.session_state.models[selected_model]
    input_data = {}
    for col in st.session_state.X.columns:
        if col in st.session_state.label_encoders:
            options = sorted(st.session_state.label_encoders[col].classes_)
            choice = st.selectbox(f"{col}", options)
            input_data[col] = st.session_state.label_encoders[col].transform([choice])[0]
        else:
            input_data[col] = st.number_input(f"{col}", float(st.session_state.df[col].min()), float(st.session_state.df[col].max()), float(st.session_state.df[col].mean()))

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    confidence = np.max(probs) * 100
    label = st.session_state.label_encoders['Injury Severity'].inverse_transform([prediction])[0] if 'Injury Severity' in st.session_state.label_encoders else prediction
    st.success(f"Predicted Injury Severity: {label}")
    st.info(f"Confidence: {confidence:.2f}%")

# Reports
elif page == "Reports":
    st.title("Dataset Summary")
    st.dataframe(st.session_state.df.describe())

# Admin Upload
elif page == "Admin Upload":
    st.title("Upload New Dataset")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if st.button("Update Dataset") and uploaded_file:
        df, X, y, X_train, X_test, y_train, y_test, label_encoders = load_data(uploaded_file)
        models, scores_df = train_models()
        st.session_state.update(locals())
        st.success("Dataset updated and models retrained!")

# Help
elif page == "Help":
    st.title("Help Guide")
    st.markdown("""
    - **Home**: View project summary and dataset preview.
    - **Data Analysis**: See visual insights, map, and model comparisons.
    - **Prediction**: Input your custom data to get severity predictions.
    - **Reports**: Explore dataset statistics.
    - **Admin Upload**: Upload and reload a new dataset.
    """)
