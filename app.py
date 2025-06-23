# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tempfile
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
Traffic accidents are a major problem worldwide, causing fatalities, property damage, and productivity loss. 
This tool predicts accident severity based on weather, road conditions, vehicle types, and driver factors.
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

# --- Data Processing Functions ---
def normalize_categories(df, custom_mappings=None):
    default_mappings = {
        'Weather Condition': {
            'Raining': 'Rain', 'Rainy': 'Rain', 'Drizzling': 'Rain',
            'Sun': 'Sunny', 'Clear': 'Sunny', 'Foggy': 'Fog', 'Overcast': 'Cloudy'
        },
        'Road Condition': {
            'Wet': 'Wet', 'Dry': 'Dry', 'Snowy': 'Snow/Ice',
            'Snow/Ice': 'Snow/Ice', 'Icy': 'Snow/Ice'
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
        df['Location_Original'] = df[location_col]
        try:
            # Extract coordinates from (lat, long) format
            coords = df[location_col].str.extract(r'\(([^,]+),\s*([^)]+)\)')
            if coords.shape[1] == 2:
                df['latitude'] = pd.to_numeric(coords[0], errors='coerce')
                df['longitude'] = pd.to_numeric(coords[1], errors='coerce')
        except Exception as e:
            st.warning(f"Location parsing error: {str(e)}")
    
    # Identify target column
    target_col = st.session_state.target_col
    if target_col not in df.columns:
        possible_targets = [col for col in df.columns if 'severity' in col.lower() or 'injury' in col.lower()]
        if possible_targets:
            target_col = possible_targets[0]
            st.session_state.target_col = target_col
    
    # Normalize and encode
    df = normalize_categories(df, custom_mappings={})
    
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.title()
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Scale numeric features
    numeric_cols = df.select_dtypes(include='number').columns.difference([target_col])
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, label_encoders, target_col

def prepare_model_data(df, target_col):
    X = df.drop([target_col], axis=1)
    loc_cols = [col for col in X.columns if 'location' in col.lower() or 'latitude' in col.lower() or 'longitude' in col.lower()]
    if loc_cols:
        X = X.drop(loc_cols, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

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
            scores = [
                name,
                accuracy_score(y_test, y_pred) * 100,
                precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
                recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
                f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            ]
            trained_models[name] = model
            model_scores.append(scores)
        except Exception as e:
            st.warning(f"Failed to train {name}: {str(e)}")
    
    return trained_models, pd.DataFrame(model_scores, 
                                      columns=['Model', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'])

# --- Initialize Data ---
if st.session_state.current_df is None:
    df, label_encoders, target_col = load_default_data()
    X_train, X_test, y_train, y_test = prepare_model_data(df, target_col)
    models, scores_df = train_models(X_train, y_train, X_test, y_test)
    
    st.session_state.update({
        'current_df': df,
        'label_encoders': label_encoders,
        'models': models,
        'scores_df': scores_df,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    })

# --- Admin Functions ---
def handle_dataset_upload(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        new_df = pd.read_csv(tmp_path)
        os.unlink(tmp_path)
        
        new_df, new_label_encoders, new_target_col = preprocess_data(new_df)
        new_X_train, new_X_test, new_y_train, new_y_test = prepare_model_data(new_df, new_target_col)
        new_models, new_scores_df = train_models(new_X_train, new_y_train, new_X_test, new_y_test)
        
        st.session_state.update({
            'current_df': new_df,
            'label_encoders': new_label_encoders,
            'models': new_models,
            'scores_df': new_scores_df,
            'X_train': new_X_train,
            'X_test': new_X_test,
            'y_train': new_y_train,
            'y_test': new_y_test,
            'target_col': new_target_col
        })
        st.success("Dataset updated successfully!")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# --- UI Components ---
st.sidebar.title("Navigation")
st.session_state.admin_mode = st.sidebar.checkbox("Admin Mode")

pages = ["Home", "Data Analysis", "Prediction", "Reports"]
if st.session_state.admin_mode:
    pages.append("Admin")
page = st.sidebar.radio("Go to", pages)

# --- Page Rendering ---
if page == "Home":
    st.title("Traffic Accident Severity Prediction")
    st.write(PROJECT_OVERVIEW)
    
    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.current_df.head())
    
    st.subheader("Dataset Statistics")
    st.write(f"Records: {len(st.session_state.current_df)}")
    st.write(f"Target: {st.session_state.target_col}")
    st.write(f"Features: {', '.join(st.session_state.X_train.columns)}")

elif page == "Data Analysis":
    st.title("Data Analysis")
    df = st.session_state.current_df
    
    st.subheader("Target Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=st.session_state.target_col, data=df, ax=ax, palette=PALETTE)
    st.pyplot(fig)
    
    st.subheader("Accident Hotspots")
    if 'latitude' in df.columns and 'longitude' in df.columns:
        coords = df[['latitude', 'longitude']].dropna()
        if len(coords) > 1000:
            coords = coords.sample(1000, random_state=42)
        st.map(coords)
    else:
        st.warning("No location data available")
    
    st.subheader("Feature Correlations")
    numeric_cols = df.select_dtypes(include='number').columns
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    if not st.session_state.scores_df.empty:
        st.subheader("Model Performance")
        st.dataframe(st.session_state.scores_df.style.format("{:.2f}"))
        
        st.subheader("Feature Importance")
        model_name = st.selectbox("Select Model", list(st.session_state.models.keys()))
        model = st.session_state.models[model_name]
        
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                importances = np.mean(np.abs(model.coefs_[0]), axis=1) if hasattr(model, 'coefs_') else None
            
            if importances is not None:
                importance_df = pd.DataFrame({
                    'Feature': st.session_state.X_train.columns,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(10)
                
                fig, ax = plt.subplots()
                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, palette=PALETTE)
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not show feature importance: {str(e)}")

elif page == "Prediction":
    st.title("Severity Prediction")
    
    if st.session_state.models:
        model_name = st.selectbox("Select Model", list(st.session_state.models.keys()))
        model = st.session_state.models[model_name]
        
        inputs = {}
        for col in st.session_state.X_train.columns:
            if col in st.session_state.label_encoders:
                options = st.session_state.label_encoders[col].classes_
                inputs[col] = st.session_state.label_encoders[col].transform(
                    [st.selectbox(col, options)])[0]
            else:
                col_data = st.session_state.current_df[col]
                inputs[col] = st.number_input(
                    col, float(col_data.min()), float(col_data.max()), float(col_data.mean()))
        
        if st.button("Predict"):
            input_df = pd.DataFrame([inputs])
            try:
                pred = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0]
                confidence = proba.max() * 100
                
                if st.session_state.target_col in st.session_state.label_encoders:
                    pred_label = st.session_state.label_encoders[st.session_state.target_col].inverse_transform([pred])[0]
                else:
                    pred_label = pred
                
                st.success(f"Predicted Severity: {pred_label} (Confidence: {confidence:.1f}%)")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

elif page == "Reports":
    st.title("Dataset Reports")
    df = st.session_state.current_df
    
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())
    
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    
    st.subheader("Data Types")
    st.write(df.dtypes)

elif page == "Admin":
    st.title("Admin Dashboard")
    password = st.text_input("Password", type="password")
    
    if password == "admin1":
        st.subheader("Upload New Dataset")
        uploaded_file = st.file_uploader("CSV File", type="csv")
        
        if uploaded_file and st.button("Update Dataset"):
            handle_dataset_upload(uploaded_file)
        
        st.subheader("System Reset")
        if st.button("Reset to Default"):
            df, label_encoders, target_col = load_default_data()
            X_train, X_test, y_train, y_test = prepare_model_data(df, target_col)
            models, scores_df = train_models(X_train, y_train, X_test, y_test)
            
            st.session_state.update({
                'current_df': df,
                'label_encoders': label_encoders,
                'models': models,
                'scores_df': scores_df,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'target_col': target_col
            })
            st.success("System reset complete!")
    else:
        st.error("Incorrect password")
