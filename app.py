# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# --- Config ---
st.set_page_config(page_title="Accident Severity Predictor", layout="wide")
# Set dark theme for all visualizations
plt.style.use('dark_background')
sns.set_style("darkgrid")
PALETTE = sns.color_palette("husl")

# --- Project Overview ---
PROJECT_OVERVIEW = """
Traffic accidents are a major problem worldwide, causing several fatalities, damage to property, and loss of productivity. Predicting accident severity based on contributors such as weather conditions, road conditions, types of vehicles, and drivers enables the authorities to take necessary actions to minimize the risk and develop better emergency responses. 
 
This project uses machine learning techniques to analyze past traffic data for accident severity prediction and present useful data to improve road safety and management.
"""

# --- Normalize Text Values ---
def normalize_categories(df):
    mappings = {
        'Weather Condition': {
            'Raining': 'Rain',
            'Rainy': 'Rain',
            'Drizzling': 'Rain',
            'Sun': 'Sunny',
            'Clear': 'Sunny',
            'Foggy': 'Fog',
            'Overcast': 'Cloudy'
        },
        'Road Condition': {
            'Wet': 'Wet',
            'Dry': 'Dry',
            'Snowy': 'Snow/Ice',
            'Snow/Ice': 'Snow/Ice',
            'Icy': 'Snow/Ice'
        },
        'Light Condition': {
            'Dark - No Street Lights': 'Dark',
            'Dark - Street Lights Off': 'Dark',
            'Dark - Street Lights On': 'Dark',
            'Daylight': 'Daylight'
        },
        # Add more column mappings as needed
    }

    for col, replacements in mappings.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col] = df[col].replace(replacements)

    return df

# --- Load Dataset ---
@st.cache_data(persist="disk")
def load_data(uploaded_file=None):
    if uploaded_file is None:
        url = 'https://raw.githubusercontent.com/narmakathir/accident-severity-streamlit/main/filtered_crash_data.csv'
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(uploaded_file)
    
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    # Extract coordinates from Location if available
    if 'Location' in df.columns:
        try:
            coords = df['Location'].str.extract(r'\(([^,]+),\s*([^)]+)\)')
            if not coords.empty:
                df['latitude'] = pd.to_numeric(coords[0], errors='coerce')
                df['longitude'] = pd.to_numeric(coords[1], errors='coerce')
                df.dropna(subset=['latitude', 'longitude'], inplace=True)
        except:
            pass

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

    X = df.drop([target_col, 'Location'], axis=1) if 'Location' in df.columns else df.drop([target_col], axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return df, X, y, X_train, X_test, y_train, y_test, label_encoders

# Initialize session state for uploaded file
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# --- Train Models ---
@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
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

# Load data based on uploaded file or default
df, X, y, X_train, X_test, y_train, y_test, label_encoders = load_data(st.session_state.uploaded_file)
models, scores_df = train_models(X_train, y_train, X_test, y_test)

# --- Side Menu ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Prediction", "Reports", "Admin"])

# --- Home ---
if page == "Home":
    st.title("Traffic Accident Severity Prediction")
    st.write(PROJECT_OVERVIEW)

    st.subheader("Dataset Preview")
    st.dataframe(df.copy().head())

# --- Data Analysis ---
elif page == "Data Analysis":
    st.title("Data Analysis & Insights")
    st.markdown("*Explore key patterns and model performance.*")
    st.divider()

    st.subheader("➥ Injury Severity Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Injury Severity', data=df, ax=ax, palette=PALETTE)
    ax.set_title('Count of Injury Levels', color='white')
    ax.set_xlabel('Injury Severity', color='white')
    ax.set_ylabel('Count', color='white')
    ax.tick_params(colors='white')
    st.pyplot(fig)
    st.divider()

    st.subheader("➥ Hotspot Location")
    if 'latitude' in df.columns and 'longitude' in df.columns:
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        # Create Folium map with dark tiles
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='CartoDB dark_matter')
        
        # Add points to the map
        sample_df = df.sample(n=min(1000, len(df)), random_state=42)
        for _, row in sample_df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.9
            ).add_to(m)
        
        folium_static(m)
    else:
        st.warning("Geographic coordinates not found in the dataset.")
    st.divider()

    st.subheader("➥ Correlation Heatmap")
    corr = df.select_dtypes(['number']).corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax)
    ax.set_title("Correlation Heatmap", color='white')
    st.pyplot(fig)
    st.divider()

    st.subheader("➥ Model Performance")
    st.table(scores_df.round(2))
    st.divider()

    st.subheader("➥ Model Comparison Bar Chart")
    performance_df = scores_df.set_index('Model')
    fig, ax = plt.subplots(figsize=(10, 6))
    performance_df.plot(kind='bar', ax=ax, color=PALETTE)
    ax.set_title('Model Comparison', color='white')
    ax.set_ylabel('Score (%)', color='white')
    ax.set_xlabel('Model', color='white')
    ax.tick_params(colors='white')
    ax.legend(title='Metrics', title_fontsize='12', fontsize='10', facecolor='black', edgecolor='white')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)
    st.divider()

    st.subheader("➥ Model-Specific Feature Importances")
    model_name = st.selectbox("Select Model", list(models.keys()), index=1)

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

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_vals, y=top_features, ax=ax, palette=PALETTE)
    ax.set_title(f'{model_name} Top 10 Features', color='white')
    ax.set_xlabel('Importance', color='white')
    ax.set_ylabel('Feature', color='white')
    ax.tick_params(colors='white')
    st.pyplot(fig)

# --- Prediction ---
elif page == "Prediction":
    st.title("Custom Prediction")
    selected_model = st.selectbox("Choose Model for Prediction", list(models.keys()))
    model = models[selected_model]

    input_data = {}
    cols_per_row = 3
    cols = st.columns(cols_per_row)
    
    for i, col in enumerate(X.columns):
        with cols[i % cols_per_row]:
            if col in label_encoders:
                options = sorted(label_encoders[col].classes_)
                choice = st.selectbox(f"{col}", options)
                input_data[col] = label_encoders[col].transform([choice])[0]
            else:
                input_data[col] = st.number_input(
                    f"{col}", 
                    float(df[col].min()), 
                    float(df[col].max()), 
                    float(df[col].mean())
    
    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        confidence = np.max(probs) * 100

        if 'Injury Severity' in label_encoders:
            severity_label = label_encoders['Injury Severity'].inverse_transform([prediction])[0]
        else:
            severity_label = prediction

        st.success(f"**Predicted Injury Severity:** {severity_label}")
        st.info(f"**Confidence:** {confidence:.2f}%")

# --- Reports ---
elif page == "Reports":
    st.title("Generated Reports")
    st.write("### Dataset Summary")
    st.dataframe(df.describe())
    
    st.write("### Missing Values Report")
    missing_values = df.isnull().sum()
    st.dataframe(missing_values[missing_values > 0].to_frame(name="Missing Values"))
    
    st.write("### Data Types")
    st.dataframe(df.dtypes.to_frame(name="Data Type"))

# --- Admin ---
elif page == "Admin":
    st.title("Admin Panel")
    st.warning("This section is for administrators only.")
    
    uploaded_file = st.file_uploader("Upload new dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.success("New dataset uploaded successfully!")
        st.info("Please refresh the page or navigate to another section to see updates.")
        
        if st.button("Clear Cache and Reload Data"):
            # Clear all caches
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
