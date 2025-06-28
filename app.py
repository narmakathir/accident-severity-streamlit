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

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    .css-1d391kg { background-color: #0F131D !important; border-right: 1px solid #2A3459; }
    .stButton>button { background-color: #1E2130; color: white; border-color: #2A3459; width: 100%; margin: 5px 0; }
    .stButton>button:hover { background-color: #2A3459; color: white; }
    .stButton>button[kind="primary"] { background-color: #3A4D8F !important; font-weight: 500; }
    .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak, .st-al, .st-am, .st-an, .st-ao, .st-ap, .st-aq, .st-ar, .st-as { 
        background-color: #1E2130; color: white; border-color: #2A3459; }
    .stTextInput input, .stSelectbox select, .stNumberInput input { color: white !important; }
    .stDataFrame { background-color: #1E2130; }
    table { color: white !important; }
    .stMarkdown { color: white; }
    hr { border-color: #2A3459; }
    .card { background-color: #1E2130; border-radius: 8px; padding: 15px; margin-bottom: 15px; border: 1px solid #2A3459; }
    .card-title { font-size: 1.2em; font-weight: bold; margin-bottom: 10px; color: #4A8DF8; }
    .nav-button { background-color: #1E2130; color: white; border: 1px solid #2A3459; border-radius: 4px; 
                 padding: 10px 15px; margin: 5px 0; width: 100%; text-align: center; cursor: pointer; transition: all 0.3s; }
    .nav-button:hover { background-color: #2A3459; }
    .nav-button.active { background-color: #3A4D8F; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Project Overview ---
PROJECT_OVERVIEW = """
<div class="card">
    <div class="card-title">Project Overview</div>
    <p>This application predicts traffic accident severity using machine learning models.</p>
</div>
"""

# --- Session State Initialization ---
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scores_df' not in st.session_state:
    st.session_state.scores_df = pd.DataFrame()
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'default_dataset' not in st.session_state:
    st.session_state.default_dataset = 'https://raw.githubusercontent.com/narmakathir/accident-severity-streamlit/main/filtered_crash_data.csv'
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'is_default_data' not in st.session_state:
    st.session_state.is_default_data = True

# --- Data Loading and Preprocessing ---
@st.cache_data(persist="disk")
def load_default_data():
    url = st.session_state.default_dataset
    df = pd.read_csv(url)
    return preprocess_data(df)

def preprocess_data(df):
    # Basic preprocessing
    df = df.copy()
    df.drop_duplicates(inplace=True)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    categorical_cols = df.select_dtypes(exclude=np.number).columns
    for col in categorical_cols:
        if col != 'Location':
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Extract coordinates
    if 'Location' in df.columns:
        df['Location'] = df['Location'].astype(str)
        location = df['Location'].str.replace(r'[()]', '', regex=True).str.split(',', expand=True)
        df['latitude'] = location[0].astype(float)
        df['longitude'] = location[1].astype(float)

    # Detect target column (flexible naming)
    possible_targets = [col for col in df.columns if 'severity' in col.lower() or 'injury' in col.lower()]
    target_col = possible_targets[0] if possible_targets else None
    st.session_state.target_col = target_col

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col != 'Location':
            df[col] = df[col].astype(str).str.strip().str.title()
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Scale numeric features (excluding target)
    if target_col and target_col in df.columns:
        numeric_cols = df.select_dtypes(include='number').columns.difference([target_col])
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, label_encoders, target_col

def prepare_model_data(df, target_col, apply_smote=False):
    if not target_col or target_col not in df.columns:
        st.error("Target column not found in dataset")
        return None, None, None, None, None, None
    
    X = df.drop([target_col, 'Location'], axis=1, errors='ignore')
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Apply SMOTE only if specified
    if apply_smote and st.session_state.is_default_data:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    return X, y, X_train, X_test, y_train, y_test

# --- Model Training ---
@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=42)
    }
    
    trained_models = {}
    model_scores = []
    model_metrics = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred) * 100
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100

            trained_models[name] = model
            model_scores.append([name, acc, prec, rec, f1])
            
            # Store metrics
            model_metrics[name] = {
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
        except Exception as e:
            st.warning(f"Failed to train {name}: {str(e)}")
            continue

    scores_df = pd.DataFrame(model_scores, columns=['Model', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'])
    return trained_models, scores_df, model_metrics

# --- Initialize with Default Data ---
if st.session_state.current_df is None:
    df, label_encoders, target_col = load_default_data()
    if target_col and target_col in df.columns:
        X, y, X_train, X_test, y_train, y_test = prepare_model_data(df, target_col, apply_smote=True)
        models, scores_df, model_metrics = train_models(X_train, y_train, X_test, y_test)
    else:
        st.error("Could not identify target column in dataset")
        models, scores_df, model_metrics = {}, pd.DataFrame(), {}

    st.session_state.current_df = df
    st.session_state.label_encoders = label_encoders
    st.session_state.models = models
    st.session_state.scores_df = scores_df
    st.session_state.model_metrics = model_metrics
    st.session_state.X = X if 'X' in locals() else None
    st.session_state.y = y if 'y' in locals() else None
    st.session_state.X_train = X_train if 'X_train' in locals() else None
    st.session_state.X_test = X_test if 'X_test' in locals() else None
    st.session_state.y_train = y_train if 'y_train' in locals() else None
    st.session_state.y_test = y_test if 'y_test' in locals() else None
    st.session_state.is_default_data = True

# --- Page Rendering Functions ---
def render_home():
    st.title("Traffic Accident Severity Prediction")
    st.markdown(PROJECT_OVERVIEW, unsafe_allow_html=True)

    with st.expander("Dataset Preview", expanded=True):
        st.dataframe(st.session_state.current_df.head())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(st.session_state.current_df))
    with col2:
        st.metric("Features Available", len(st.session_state.current_df.columns))
    with col3:
        st.metric("Target Variable", st.session_state.target_col or "Not detected")

def render_data_analysis():
    st.title("Data Analysis & Insights")
    df = st.session_state.current_df
    target_col = st.session_state.target_col

    # Target Variable Distribution
    with st.expander("Target Variable Distribution", expanded=True):
        if target_col and target_col in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x=target_col, data=df, ax=ax, palette="coolwarm")
            
            # Auto-generate labels based on unique values
            unique_values = sorted(df[target_col].unique())
            if len(unique_values) <= 10:  # Only label if reasonable number of categories
                ax.set_xticklabels([f"Level {val}" for val in unique_values], rotation=45, ha='right')
            
            ax.set_title(f'Distribution of {target_col}', color='white')
            ax.set_xlabel('Severity Level', color='white')
            ax.set_ylabel('Count', color='white')
            st.pyplot(fig)
        else:
            st.warning(f"Target column '{target_col}' not found in dataset")

    # Accident Hotspot Locations
    with st.expander("Accident Hotspot Locations"):
        if 'latitude' in df.columns and 'longitude' in df.columns:
            m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], 
                         zoom_start=11, tiles='CartoDB dark_matter')
            
            sample_df = df.sample(min(1000, len(df)), random_state=42)
            for _, row in sample_df.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.9
                ).add_to(m)
            
            folium_static(m, width=1000, height=600)
        else:
            st.warning("Location data not available for mapping")

    # Feature Correlation Heatmap
    with st.expander("Feature Correlation Heatmap"):
        if target_col and target_col in df.columns:
            try:
                numeric_cols = df.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 1:  # Need at least 2 numeric columns for correlation
                    corr = df[numeric_cols].corr()
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax, center=0,
                              cbar_kws={'label': 'Correlation Coefficient'})
                    ax.set_title("Feature Correlation Heatmap", color='white', pad=20)
                    st.pyplot(fig)
                else:
                    st.warning("Not enough numeric features for correlation analysis")
            except Exception as e:
                st.warning(f"Could not generate correlation heatmap: {str(e)}")
        else:
            st.warning("Target column not available for correlation analysis")

    # Model Performance Metrics
    if not st.session_state.scores_df.empty:
        with st.expander("Model Performance Metrics"):
            st.dataframe(st.session_state.scores_df.style.format({
                'Accuracy (%)': '{:.2f}',
                'Precision (%)': '{:.2f}',
                'Recall (%)': '{:.2f}',
                'F1-Score (%)': '{:.2f}'
            }))

            # Model Comparison Chart
            fig, ax = plt.subplots(figsize=(10, 6))
            st.session_state.scores_df.set_index('Model').plot(kind='bar', ax=ax, cmap='coolwarm')
            ax.set_title('Model Performance Comparison', color='white')
            ax.set_ylabel('Score (%)', color='white')
            ax.set_xlabel('Model', color='white')
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

    # Feature Importance Analysis
    with st.expander("Feature Importance Analysis"):
        if st.session_state.models and st.session_state.X is not None:
            model_name = st.selectbox("Select Model", list(st.session_state.models.keys()))
            model = st.session_state.models[model_name]
            
            try:
                if hasattr(model, 'feature_importances_'):  # Random Forest, XGBoost
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):  # Logistic Regression
                    importances = np.abs(model.coef_[0])
                elif hasattr(model, 'coefs_'):  # Neural Network
                    importances = np.mean(np.abs(model.coefs_[0]), axis=1)
                else:
                    raise AttributeError("Model doesn't support feature importance")
                
                # Normalize importances
                importances = importances / importances.sum()
                sorted_idx = np.argsort(importances)[::-1]
                top_features = st.session_state.X.columns[sorted_idx][:10]
                top_importances = importances[sorted_idx][:10]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=top_importances, y=top_features, ax=ax, palette="coolwarm")
                ax.set_title(f'{model_name} - Feature Importance', color='white')
                ax.set_xlabel('Importance Score', color='white')
                ax.set_ylabel('Feature', color='white')
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not display feature importances: {str(e)}")

def render_prediction():
    st.title("Accident Severity Prediction")
    
    if not st.session_state.models or st.session_state.X is None:
        st.warning("No trained models available. Please check the Data Analysis page.")
        return
    
    st.subheader("Model Selection")
    model_name = st.selectbox("Select Model", list(st.session_state.models.keys()))
    model = st.session_state.models[model_name]
    
    st.subheader("Input Parameters")
    input_data = {}
    cols = st.columns(2)
    
    for i, feature in enumerate(st.session_state.X.columns):
        col = cols[i % 2]
        if feature in st.session_state.label_encoders:
            options = st.session_state.label_encoders[feature].classes_
            input_data[feature] = col.selectbox(feature, options)
            input_data[feature] = st.session_state.label_encoders[feature].transform([input_data[feature]])[0]
        else:
            min_val = st.session_state.current_df[feature].min()
            max_val = st.session_state.current_df[feature].max()
            default_val = st.session_state.current_df[feature].median()
            input_data[feature] = col.number_input(feature, min_val, max_val, default_val)
    
    if st.button("Predict Severity"):
        input_df = pd.DataFrame([input_data])
        try:
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            
            if st.session_state.target_col in st.session_state.label_encoders:
                severity_labels = st.session_state.label_encoders[st.session_state.target_col].classes_
                prediction_label = severity_labels[prediction]
            else:
                prediction_label = str(prediction)
            
            confidence = np.max(proba) * 100
            
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Severity", prediction_label)
            with col2:
                st.metric("Confidence", f"{confidence:.2f}%")
            
            if st.session_state.target_col in st.session_state.label_encoders:
                st.subheader("Probability Distribution")
                prob_df = pd.DataFrame({
                    'Severity Level': severity_labels,
                    'Probability': proba * 100
                })
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(x='Severity Level', y='Probability', data=prob_df, ax=ax, palette="coolwarm")
                ax.set_title('Prediction Probability Distribution', color='white')
                ax.set_xlabel('Severity Level', color='white')
                ax.set_ylabel('Probability (%)', color='white')
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

def render_reports():
    st.title("Dataset Reports")
    
    with st.expander("Dataset Summary", expanded=True):
        st.dataframe(st.session_state.current_df.describe())
    
    with st.expander("Column Information"):
        col_info = pd.DataFrame({
            'Column': st.session_state.current_df.columns,
            'Data Type': st.session_state.current_df.dtypes,
            'Missing Values': st.session_state.current_df.isna().sum(),
            'Unique Values': st.session_state.current_df.nunique()
        })
        st.dataframe(col_info)

def render_help():
    st.title("User Guide")
    
    with st.expander("How to Use This App"):
        st.markdown("""
        - **Home**: Overview of the dataset
        - **Data Analysis**: Explore visualizations and model performance
        - **Prediction**: Make custom predictions
        - **Reports**: View detailed dataset information
        """)
    
    with st.expander("About the Models"):
        st.markdown("""
        The app uses four machine learning models:
        1. Logistic Regression
        2. Random Forest
        3. XGBoost
        4. Neural Network
        """)

def render_admin():
    st.title("Admin Dashboard")
    password = st.text_input("Enter Admin Password:", type="password")
    
    if password != "admin123":
        st.error("Incorrect password")
        return
    
    st.warning("Admin mode activated")
    
    with st.expander("Upload New Dataset"):
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                st.success("Dataset uploaded successfully")
                
                if st.button("Update System"):
                    with st.spinner("Processing..."):
                        new_df, new_encoders, new_target = preprocess_data(new_df)
                        if new_target:
                            X, y, X_train, X_test, y_train, y_test = prepare_model_data(new_df, new_target, False)
                            models, scores, metrics = train_models(X_train, y_train, X_test, y_test)
                            
                            # Update session state
                            st.session_state.current_df = new_df
                            st.session_state.label_encoders = new_encoders
                            st.session_state.models = models
                            st.session_state.scores_df = scores
                            st.session_state.model_metrics = metrics
                            st.session_state.X = X
                            st.session_state.y = y
                            st.session_state.target_col = new_target
                            st.session_state.is_default_data = False
                            
                            st.success("System updated with new dataset!")
                        else:
                            st.error("Could not identify target column in new dataset")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# --- Sidebar Navigation ---
def create_sidebar():
    st.sidebar.title("Navigation")
    pages = ["Home", "Data Analysis", "Prediction", "Reports", "Help"]
    
    if st.sidebar.checkbox("Admin Mode"):
        pages.append("Admin")
    
    for page in pages:
        if st.sidebar.button(page):
            st.session_state.current_page = page
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Current Dataset:** {len(st.session_state.current_df) if st.session_state.current_df is not None else 0} records")
    st.sidebar.markdown(f"**Target Variable:** {st.session_state.target_col or 'Not set'}")

# --- Main App ---
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
