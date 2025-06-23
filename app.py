# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tempfile
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
sns.set_style("whitegrid")
PALETTE = sns.color_palette("Set2")

# --- Project Overview ---
PROJECT_OVERVIEW = """
Traffic accidents are a major problem worldwide, causing several fatalities, damage to property, and loss of productivity. 
Predicting accident severity based on contributors such as weather conditions, road conditions, types of vehicles, and drivers 
enables the authorities to take necessary actions to minimize the risk and develop better emergency responses.
"""

# --- Session State ---
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scores_df' not in st.session_state:
    st.session_state.scores_df = pd.DataFrame()
if 'target_col' not in st.session_state:
    st.session_state.target_col = 'Injury Severity'
if 'default_dataset' not in st.session_state:
    st.session_state.default_dataset = 'https://raw.githubusercontent.com/narmakathir/accident-severity-streamlit/main/filtered_crash_data.csv'

# --- Helper Functions ---
def normalize_categories(df, custom_mappings=None):
    default_mappings = {
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
    }
    mappings = custom_mappings if custom_mappings else default_mappings
    
    for col, replacements in mappings.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col] = df[col].replace(replacements)
    return df

@st.cache_data(persist="disk")
def load_default_data():
    url = st.session_state.default_dataset
    df = pd.read_csv(url)
    return preprocess_data(df)

def preprocess_data(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    
    # Handle location data
    location_col = next((col for col in df.columns if 'location' in col.lower()), None)
    if location_col:
        df['Location'] = df[location_col].astype(str)
    
    target_col = st.session_state.target_col
    if target_col not in df.columns:
        possible_targets = [col for col in df.columns if 'severity' in col.lower() or 'injury' in col.lower()]
        if possible_targets:
            target_col = possible_targets[0]
            st.session_state.target_col = target_col
    
    df = normalize_categories(df, custom_mappings={})
    
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.title()
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    numeric_cols = df.select_dtypes(include='number').columns.difference([target_col])
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, label_encoders, target_col

def prepare_model_data(df, target_col):
    X = df.drop([target_col], axis=1)
    loc_cols = [col for col in X.columns if 'location' in col.lower()]
    if loc_cols:
        X = X.drop(loc_cols, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X, y, X_train, X_test, y_train, y_test

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
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            trained_models[name] = model
            model_scores.append([
                name,
                accuracy_score(y_test, y_pred) * 100,
                precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
                recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
                f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            ])
        except Exception as e:
            st.warning(f"Failed to train {name}: {str(e)}")
    
    scores_df = pd.DataFrame(model_scores, 
                           columns=['Model', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'])
    return trained_models, scores_df

# --- Initialize Data ---
if st.session_state.current_df is None:
    df, label_encoders, target_col = load_default_data()
    X, y, X_train, X_test, y_train, y_test = prepare_model_data(df, target_col)
    models, scores_df = train_models(X_train, y_train, X_test, y_test)
    
    st.session_state.current_df = df
    st.session_state.label_encoders = label_encoders
    st.session_state.models = models
    st.session_state.scores_df = scores_df
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

# --- Pages ---
st.sidebar.title("Navigation")
if 'admin_mode' not in st.session_state:
    st.session_state.admin_mode = False
st.session_state.admin_mode = st.sidebar.checkbox("Admin Mode")

pages = ["Home", "Data Analysis", "Prediction", "Reports"]
if st.session_state.admin_mode:
    pages.append("Admin")
page = st.sidebar.radio("Go to", pages)

# --- Home Page ---
if page == "Home":
    st.title("Traffic Accident Severity Prediction")
    st.write(PROJECT_OVERVIEW)
    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.current_df.head())
    st.subheader("Current Dataset Info")
    st.write(f"Records: {len(st.session_state.current_df)}")
    st.write(f"Target: {st.session_state.target_col}")
    st.write(f"Features: {', '.join(st.session_state.X.columns)}")

# --- Data Analysis Page ---
elif page == "Data Analysis":
    st.title("Data Analysis & Insights")
    df = st.session_state.current_df
    
    # Target Distribution
    st.subheader("➥ Target Variable Distribution")
    if st.session_state.target_col in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x=st.session_state.target_col, data=df, ax=ax, palette=PALETTE)
        st.pyplot(fig)
    st.divider()
    
    # Location Mapping
    st.subheader("➥ Hotspot Location")
    if 'Location' in df.columns:
        try:
            # Extract coordinates
            locations = df['Location'].astype(str).str.strip()
            coords = locations.str.extract(r'\(([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+)\)')
            coords.columns = ['latitude', 'longitude']
            coords = coords.apply(pd.to_numeric, errors='coerce').dropna()
            
            if not coords.empty:
                # Create map
                m = folium.Map(
                    location=[coords['latitude'].mean(), coords['longitude'].mean()],
                    zoom_start=11,
                    tiles='CartoDB dark_matter'
                )
                
                # Add markers
                for idx, row in coords.sample(min(1000, len(coords))).iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=5,
                        color='red',
                        fill=True,
                        fill_opacity=0.7
                    ).add_to(m)
                
                folium_static(m, width=1000, height=600)
                st.success(f"Mapped {len(coords)} locations")
            else:
                st.error("No valid coordinates found. Check format matches '(lat, long)'")
        except Exception as e:
            st.error(f"Location processing failed: {str(e)}")
    else:
        st.warning("No location data found")
    st.divider()
    
    # Correlation Heatmap
    st.subheader("➥ Correlation Heatmap")
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.select_dtypes(['number']).corr(), cmap='YlGnBu', ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not generate heatmap: {str(e)}")
    st.divider()
    
    # Model Performance
    if not st.session_state.scores_df.empty:
        st.subheader("➥ Model Performance")
        st.dataframe(st.session_state.scores_df.style.format("{:.2f}"))
        st.divider()
        
        st.subheader("➥ Model Comparison")
        fig, ax = plt.subplots()
        st.session_state.scores_df.set_index('Model').plot(kind='bar', ax=ax)
        st.pyplot(fig)
        st.divider()

# --- Prediction Page ---
elif page == "Prediction":
    st.title("Custom Prediction")
    if not st.session_state.models:
        st.warning("No trained models available")
    else:
        model = st.session_state.models[st.selectbox("Model", list(st.session_state.models.keys()))]
        
        input_data = {}
        for col in st.session_state.X.columns:
            if col in st.session_state.label_encoders:
                options = st.session_state.label_encoders[col].classes_
                input_data[col] = st.session_state.label_encoders[col].transform(
                    [st.selectbox(col, options)])[0]
            else:
                input_data[col] = st.number_input(
                    col,
                    float(st.session_state.current_df[col].min()),
                    float(st.session_state.current_df[col].max()),
                    float(st.session_state.current_df[col].mean()))
        
        if st.button("Predict"):
            try:
                pred = model.predict(pd.DataFrame([input_data]))[0]
                proba = model.predict_proba(pd.DataFrame([input_data]))[0]
                confidence = proba.max() * 100
                
                if st.session_state.target_col in st.session_state.label_encoders:
                    pred_label = st.session_state.label_encoders[st.session_state.target_col].inverse_transform([pred])[0]
                else:
                    pred_label = pred
                
                st.success(f"Prediction: {pred_label} (Confidence: {confidence:.1f}%)")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# --- Reports Page ---
elif page == "Reports":
    st.title("Dataset Reports")
    df = st.session_state.current_df
    
    st.subheader("Dataset Summary")
    st.dataframe(df.describe())
    
    st.subheader("Missing Values")
    st.dataframe(df.isna().sum().rename("Missing Count"))
    
    st.subheader("Column Information")
    st.dataframe(pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Unique Values': df.nunique()
    }))

# --- Admin Page ---
elif page == "Admin":
    st.title("Admin Dashboard")
    if st.text_input("Password", type="password") != "admin123":
        st.error("Incorrect password")
        st.stop()
    
    st.subheader("Upload New Dataset")
    uploaded_file = st.file_uploader("CSV File", type="csv")
    if uploaded_file and st.button("Update Dataset"):
        try:
            df = pd.read_csv(uploaded_file)
            df, label_encoders, target_col = preprocess_data(df)
            X, y, X_train, X_test, y_train, y_test = prepare_model_data(df, target_col)
            models, scores_df = train_models(X_train, y_train, X_test, y_test)
            
            st.session_state.current_df = df
            st.session_state.label_encoders = label_encoders
            st.session_state.models = models
            st.session_state.scores_df = scores_df
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.target_col = target_col
            
            st.success("Dataset updated successfully")
        except Exception as e:
            st.error(f"Failed to update: {str(e)}")
    
    st.subheader("System Reset")
    if st.button("Reset to Default Dataset"):
        df, label_encoders, target_col = load_default_data()
        X, y, X_train, X_test, y_train, y_test = prepare_model_data(df, target_col)
        models, scores_df = train_models(X_train, y_train, X_test, y_test)
        
        st.session_state.current_df = df
        st.session_state.label_encoders = label_encoders
        st.session_state.models = models
        st.session_state.scores_df = scores_df
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.target_col = target_col
        
        st.success("System reset complete")
