# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import io

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
Traffic accidents are a major problem worldwide, causing several fatalities, damage to property, and loss of productivity. 
Predicting accident severity based on contributors such as weather conditions, road conditions, types of vehicles, and drivers 
enables the authorities to take necessary actions to minimize the risk and develop better emergency responses. 
 
This project uses machine learning techniques to analyze past traffic data for accident severity prediction and present useful 
data to improve road safety and management.
"""

# --- Normalize Text Values ---
def normalize_categories(df):
    mappings = {
        'Weather': {
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
        'Light': {
            'Dark - No Street Lights': 'Dark',
            'Dark - Street Lights Off': 'Dark',
            'Dark - Street Lights On': 'Dark',
            'Daylight': 'Daylight'
        },
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
        try:
            # Read the file content into memory
            file_content = uploaded_file.getvalue().decode('utf-8')
            
            # Check if content is empty
            if not file_content.strip():
                st.error("Uploaded file is empty")
                st.stop()
                
            # Try reading the CSV
            df = pd.read_csv(io.StringIO(file_content))
            
            # Check if dataframe is empty
            if df.empty:
                st.error("Uploaded file contains no data")
                st.stop()
                
            # Check if dataframe has no columns
            if len(df.columns) == 0:
                st.error("Uploaded file has no recognizable columns")
                st.stop()
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.stop()
    
    # Basic data cleaning
    df.drop_duplicates(inplace=True)
    
    # Handle missing values - numeric first, then categorical
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include='object').columns
    if len(categorical_cols) > 0:
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    # Extract coordinates from Location if available
    if 'Location' in df.columns:
        try:
            coords = df['Location'].str.extract(r'\(([^,]+),\s*([^)]+)\)')
            if not coords.empty:
                df['latitude'] = pd.to_numeric(coords[0], errors='coerce')
                df['longitude'] = pd.to_numeric(coords[1], errors='coerce')
                df.dropna(subset=['latitude', 'longitude'], inplace=True)
        except Exception as e:
            st.warning(f"Could not parse location data: {str(e)}")

    df = normalize_categories(df)

    # Label encoding for categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.title()
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Handle target column - flexible naming
    target_col = None
    possible_targets = ['Injury Severity', 'Severity', 'Injury', 'Accident Severity', 'Target', 'Accident_Severity']
    for possible_target in possible_targets:
        if possible_target in df.columns:
            target_col = possible_target
            break
    
    if target_col is None:
        st.warning("⚠️ Could not find standard target column. Using first numeric column as target.")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[0]
        else:
            st.error("No suitable target column found in the dataset")
            st.stop()

    # Feature scaling
    numeric_cols = df.select_dtypes(include=np.number).columns.difference([target_col])
    scaler = StandardScaler()
    if len(numeric_cols) > 0:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Prepare features - handle missing columns gracefully
    columns_to_drop = [target_col]
    if 'Location' in df.columns:
        columns_to_drop.append('Location')
    
    X = df.drop(columns=columns_to_drop, errors='ignore')
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return df, X, y, X_train, X_test, y_train, y_test, label_encoders, target_col

# Initialize session state for uploaded file
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'file_content' not in st.session_state:
    st.session_state.file_content = None

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
try:
    if st.session_state.uploaded_file is not None and st.session_state.file_content is not None:
        # Use the cached file content
        df, X, y, X_train, X_test, y_train, y_test, label_encoders, target_col = load_data(io.StringIO(st.session_state.file_content))
    else:
        # Load default or newly uploaded data
        df, X, y, X_train, X_test, y_train, y_test, label_encoders, target_col = load_data(st.session_state.uploaded_file)
    
    models, scores_df = train_models(X_train, y_train, X_test, y_test)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# --- Side Menu ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Prediction", "Reports", "Admin"])

# [Rest of your existing code for pages remains exactly the same...]

# --- Admin ---
elif page == "Admin":
    st.title("Admin Panel")
    st.warning("This section is for administrators only.")
    
    uploaded_file = st.file_uploader("Upload new dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read and store the file content in memory
            file_content = uploaded_file.getvalue().decode('utf-8')
            
            # Validate content
            if not file_content.strip():
                st.error("⚠️ Uploaded file is empty")
                st.stop()
                
            # Store in session state
            st.session_state.uploaded_file = uploaded_file
            st.session_state.file_content = file_content
            
            st.success("✅ New dataset uploaded successfully!")
            
            # Clear caches
            st.cache_data.clear()
            st.cache_resource.clear()
            
            st.info("ℹ️ Please refresh the page or navigate to another section to see updates")
            
        except UnicodeDecodeError:
            st.error("⚠️ File encoding issue - please upload a standard UTF-8 CSV file")
        except Exception as e:
            st.error(f"⚠️ Error reading uploaded file: {str(e)}")
    
    if st.button("Reset to Default Dataset"):
        st.session_state.uploaded_file = None
        st.session_state.file_content = None
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Reset to default dataset complete! Refresh the page.")
