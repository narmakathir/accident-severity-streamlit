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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# --- Custom Dark Theme Configuration ---
def set_dark_theme():
    sns.set_style("darkgrid")
    PALETTE = sns.color_palette("coolwarm")
    plt.rcParams['figure.facecolor'] = '#0E1117'
    plt.rcParams['axes.facecolor'] = '#0E1117'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['grid.color'] = '#2A3459'
    return PALETTE

PALETTE = set_dark_theme()

# --- Streamlit Config ---
st.set_page_config(page_title="Accident Severity Predictor", layout="wide")

# Custom CSS (unchanged from original)
st.markdown("""
<style>
    /* Previous CSS styles remain exactly the same */
    /* ... */
</style>
""", unsafe_allow_html=True)

# --- Project Overview ---
PROJECT_OVERVIEW = """
<div class="card">
    <div class="card-title">Project Overview</div>
    <p>Traffic accidents are a major problem worldwide, causing several fatalities, damage to property, and loss of productivity. Predicting accident severity based on contributors such as weather conditions, road conditions, types of vehicles, and drivers enables the authorities to take necessary actions to minimize the risk and develop better emergency responses.</p>
    <p>This project uses machine learning techniques to analyze past traffic data for accident severity prediction and present useful data to improve road safety and management.</p>
</div>
"""

# --- Session State for Dynamic Updates ---
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
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# --- Data Loading and Preprocessing ---
@st.cache_data(persist="disk")
def load_default_data():
    url = st.session_state.default_dataset
    df = pd.read_csv(url)
    return preprocess_data(df)

def preprocess_data(df):
    # Data Cleaning (matches PDF)
    df = df.copy()
    
    # Drop duplicates (PDF shows 4845 duplicates initially, then 141 after cleaning)
    df.drop_duplicates(inplace=True)
    
    # Handle missing values - fill numeric with median, categorical with mode (matches PDF)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    
    # Feature Engineering (matches PDF)
    # Label Encoding for categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col != 'Location':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    # Normalize numeric columns (StandardScaler as in PDF)
    target_col = 'Injury Severity'
    numeric_cols = df.select_dtypes(include='number').columns.difference([target_col])
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Extract coordinates from Location (matches PDF)
    if 'Location' in df.columns:
        location = df['Location'].str.replace(r'[()]', '', regex=True).str.split(', ', expand=True)
        df['latitude'] = location[0].astype(float)
        df['longitude'] = location[1].astype(float)
    
    return df, label_encoders, target_col

def prepare_model_data(df, target_col):
    # Feature and Target Split (matches PDF)
    X = df.drop([target_col, 'Location'], axis=1)
    y = df[target_col]
    
    # Train-Test Split (matches PDF's 80-20 split with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Apply SMOTE ONLY on Training Set (matches PDF)
    print("Before SMOTE:", Counter(y_train))
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("After SMOTE:", Counter(y_train_resampled))
    
    return X, y, X_train_resampled, X_test, y_train_resampled, y_test

# --- Model Training ---
@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),  # Matches PDF
        'Random Forest': RandomForestClassifier(random_state=42),  # Matches PDF
        'XGBoost': xgb.XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'  # Matches PDF
        ),
        'Artificial Neural Network': MLPClassifier(
            hidden_layer_sizes=(100,),  # Matches PDF
            max_iter=300,  # Matches PDF
            activation='relu',  # Matches PDF
            solver='adam',  # Matches PDF
            random_state=42
        )
    }
    
    trained_models = {}
    model_scores = []
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Metrics calculation (matches PDF)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            trained_models[name] = model
            model_scores.append([name, acc*100, prec*100, rec*100, f1*100])
            
            # Print classification report (matches PDF output format)
            print(f"\n{name} Evaluation:")
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
            
        except Exception as e:
            st.warning(f"Failed to train {name}: {str(e)}")
            continue
    
    scores_df = pd.DataFrame(model_scores, columns=['Model', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'])
    return trained_models, scores_df

# --- Initialize with Default Data ---
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

# --- Rest of the application code remains exactly the same ---
# (All the page rendering functions, navigation, etc. stay unchanged)
# Only the data loading, preprocessing, and model training were modified
# to match the PDF exactly.

# --- Navigation Functions ---
def navigate_to(page):
    st.session_state.current_page = page

# --- Normalize Text Values ---
def normalize_categories(df, custom_mappings=None):
    # ... (unchanged from original)

# --- Page Rendering Functions ---
def render_home():
    # ... (unchanged from original)

def render_data_analysis():
    # ... (unchanged from original)

def render_prediction():
    # ... (unchanged from original)

def render_reports():
    # ... (unchanged from original)

def render_help():
    # ... (unchanged from original)

def render_admin():
    # ... (unchanged from original)

def create_sidebar():
    # ... (unchanged from original)

def main():
    create_sidebar()
    
    if st.session_state.current_page == "Home":
        render_home()
    elif st.session_state.current_page == "Data Analysis":
        render_data_analysis()
    elif st.session_state.current_page == "Prediction":
        render_prediction()
    elif st.session_state.current_page == "Reports":
        render_reports()
    elif st.session_state.current_page == "Help":
        render_help()
    elif st.session_state.current_page == "Admin":
        render_admin()

if __name__ == "__main__":
    main()
