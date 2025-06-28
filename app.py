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
    .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak, .st-al, .st-am, .st-an, .st-ao, .st-ap, .st-aq, .st-ar, .st-as { background-color: #1E2130; color: white; border-color: #2A3459; }
    .stTextInput input, .stSelectbox select, .stNumberInput input { color: white !important; }
    .stDataFrame { background-color: #1E2130; }
    table { color: white !important; }
    .stMarkdown { color: white; }
    hr { border-color: #2A3459; }
    .card { background-color: #1E2130; border-radius: 8px; padding: 15px; margin-bottom: 15px; border: 1px solid #2A3459; }
    .card-title { font-size: 1.2em; font-weight: bold; margin-bottom: 10px; color: #4A8DF8; }
    .nav-button { background-color: #1E2130; color: white; border: 1px solid #2A3459; border-radius: 4px; padding: 10px 15px; margin: 5px 0; width: 100%; text-align: center; cursor: pointer; transition: all 0.3s; }
    .nav-button:hover { background-color: #2A3459; }
    .nav-button.active { background-color: #3A4D8F; font-weight: bold; }
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
    st.session_state.target_col = 'Injury Severity'
if 'default_dataset' not in st.session_state:
    st.session_state.default_dataset = 'https://raw.githubusercontent.com/narmakathir/accident-severity-streamlit/main/filtered_crash_data.csv'
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# --- Navigation Functions ---
def navigate_to(page):
    st.session_state.current_page = page

# --- Data Loading and Preprocessing ---
@st.cache_data(persist="disk")
def load_default_data():
    url = st.session_state.default_dataset
    df = pd.read_csv(url)
    return preprocess_data(df)

def preprocess_data(df):
    df = df.copy()
    
    # Basic cleaning
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    
    # Location extraction
    if 'Location' in df.columns:
        try:
            location = df['Location'].str.replace(r'[()]', '', regex=True).str.split(', ', expand=True)
            df['latitude'] = location[0].astype(float)
            df['longitude'] = location[1].astype(float)
        except:
            pass
    
    # Target column identification
    target_col = st.session_state.target_col
    if target_col not in df.columns:
        possible_targets = [col for col in df.columns if 'severity' in col.lower() or 'injury' in col.lower()]
        if possible_targets:
            target_col = possible_targets[0]
            st.session_state.target_col = target_col
    
    # Label encoding
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col != 'Location':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Feature scaling
    numeric_cols = df.select_dtypes(include='number').columns.difference([target_col])
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, label_encoders, target_col

def prepare_model_data(df, target_col):
    # Ensure numeric data only
    X = df.drop([target_col, 'Location'], axis=1, errors='ignore').select_dtypes(include=['number'])
    y = df[target_col]
    
    # Convert y to numeric
    if y.dtype == 'object':
        y = pd.to_numeric(y, errors='coerce')
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Apply SMOTE safely
    try:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        return X, y, X_train_res, X_test, y_train_res, y_test
    except Exception as e:
        st.warning(f"SMOTE application skipped: {str(e)}")
        return X, y, X_train, X_test, y_train, y_test

# --- Model Training ---
@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=42)
    }
    
    trained_models = {}
    model_scores = []
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            trained_models[name] = model
            model_scores.append([name, acc*100, prec*100, rec*100, f1*100])
        except Exception as e:
            st.warning(f"Failed to train {name}: {str(e)}")
            continue
    
    scores_df = pd.DataFrame(
        model_scores, 
        columns=['Model', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']
    )
    return trained_models, scores_df

# --- Initialize with Default Data ---
if st.session_state.current_df is None:
    try:
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
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")

# --- Page Rendering Functions ---
def render_home():
    st.title("Traffic Accident Severity Prediction")
    st.markdown(PROJECT_OVERVIEW, unsafe_allow_html=True)

    with st.expander("Dataset Preview", expanded=True):
        st.dataframe(st.session_state.current_df.head())
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(st.session_state.current_df))
    with col2:
        st.metric("Features Available", len(st.session_state.current_df.columns))
    with col3:
        st.metric("Trained Models", len(st.session_state.models))

def render_data_analysis():
    st.title("Data Analysis & Insights")
    df = st.session_state.current_df

    with st.expander("Target Variable Distribution", expanded=True):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=st.session_state.target_col, data=df, ax=ax, palette="coolwarm")
        ax.set_title(f'Distribution of {st.session_state.target_col}', color='white')
        ax.set_xlabel('Severity Level', color='white')
        ax.set_ylabel('Count', color='white')
        st.pyplot(fig)

    with st.expander("Accident Hotspot Locations"):
        if 'latitude' in df.columns and 'longitude' in df.columns:
            m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], 
                          zoom_start=11, 
                          tiles='CartoDB dark_matter')
            for _, row in df.sample(min(1000, len(df))).iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    color='#ff7f0e',
                    fill=True,
                    fill_opacity=0.7
                ).add_to(m)
            folium_static(m, width=1000, height=600)
        else:
            st.warning("No location data found")

    with st.expander("Feature Correlation Heatmap"):
        try:
            corr = df.select_dtypes(['number']).corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax, center=0)
            ax.set_title("Feature Correlation Heatmap", color='white', pad=20)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate heatmap: {str(e)}")

    if not st.session_state.scores_df.empty:
        with st.expander("Model Performance Metrics"):
            st.table(st.session_state.scores_df.style.format("{:.2f}"))
            
            st.subheader("Model Performance Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            st.session_state.scores_df.set_index('Model').plot(kind='bar', ax=ax, cmap='coolwarm')
            ax.set_title('Model Performance Comparison', color='white')
            ax.set_ylabel('Score (%)', color='white')
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
    
    with st.expander("Feature Importance Analysis"):
        model_name = st.selectbox("Select Model", list(st.session_state.models.keys()), index=1)
        model = st.session_state.models[model_name]

        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                importances = np.mean(np.abs(model.coefs_[0]), axis=1) if hasattr(model, 'coefs_') else None
            
            if importances is not None:
                importances = importances / importances.sum()
                sorted_idx = np.argsort(importances)[::-1]
                top_features = st.session_state.X.columns[sorted_idx][:10]
                top_vals = importances[sorted_idx][:10]

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=top_vals, y=top_features, ax=ax, palette="coolwarm")
                ax.set_title(f'{model_name} Feature Importance', color='white')
                st.pyplot(fig)
            else:
                st.warning("Feature importance not available for this model")
        except Exception as e:
            st.warning(f"Could not display feature importances: {str(e)}")

def render_prediction():
    st.title("Accident Severity Prediction")

    if not st.session_state.models:
        st.warning("No models available for prediction")
    else:
        model = st.session_state.models[st.selectbox("Select Model", list(st.session_state.models.keys()))]
        
        input_data = {}
        cols = st.columns(2)
        for i, col in enumerate(st.session_state.X.columns):
            with cols[i % 2]:
                if col in st.session_state.label_encoders:
                    options = st.session_state.label_encoders[col].classes_
                    choice = st.selectbox(col, options)
                    input_data[col] = st.session_state.label_encoders[col].transform([choice])[0]
                else:
                    stats = st.session_state.current_df[col].describe()
                    input_data[col] = st.number_input(
                        col, 
                        float(stats['min']), 
                        float(stats['max']), 
                        float(stats['mean'])
                    )

        if st.button("Predict Severity"):
            try:
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                probs = model.predict_proba(input_df)[0]
                
                if st.session_state.target_col in st.session_state.label_encoders:
                    severity = st.session_state.label_encoders[st.session_state.target_col].inverse_transform([prediction])[0]
                else:
                    severity = prediction
                
                st.success(f"Predicted Severity: {severity}")
                st.write(f"Confidence: {np.max(probs)*100:.2f}%")
                
                if st.session_state.target_col in st.session_state.label_encoders:
                    prob_df = pd.DataFrame({
                        'Severity': st.session_state.label_encoders[st.session_state.target_col].classes_,
                        'Probability': probs
                    })
                    st.bar_chart(prob_df.set_index('Severity'))
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

def render_reports():
    st.title("Dataset Reports")
    
    with st.expander("Summary Statistics"):
        st.dataframe(st.session_state.current_df.describe())
    
    with st.expander("Column Information"):
        col_info = pd.DataFrame({
            'Column': st.session_state.current_df.columns,
            'Type': st.session_state.current_df.dtypes,
            'Unique Values': [st.session_state.current_df[col].nunique() for col in st.session_state.current_df.columns]
        })
        st.dataframe(col_info)
    
    with st.expander("Missing Values"):
        missing = st.session_state.current_df.isnull().sum()
        st.dataframe(missing[missing > 0].rename('Missing Values'))

def render_help():
    st.title("User Guide")
    
    with st.expander("Application Overview"):
        st.markdown(PROJECT_OVERVIEW, unsafe_allow_html=True)
    
    with st.expander("Navigation Guide"):
        st.write("Use the sidebar to navigate between different sections")
    
    with st.expander("How to Use"):
        st.write("1. View data analysis on the Data Analysis page")
        st.write("2. Make predictions on the Prediction page")
        st.write("3. View dataset details on the Reports page")

def render_admin():
    st.title("Admin Dashboard")
    password = st.text_input("Enter Admin Password:", type="password")
    
    if password != "admin1":
        st.error("Incorrect password")
        return
    
    with st.expander("Dataset Management"):
        uploaded_file = st.file_uploader("Upload new dataset", type="csv")
        if uploaded_file and st.button("Update Dataset"):
            try:
                new_df = pd.read_csv(uploaded_file)
                df, label_encoders, target_col = preprocess_data(new_df)
                X, y, X_train, X_test, y_train, y_test = prepare_model_data(df, target_col)
                models, scores_df = train_models(X_train, y_train, X_test, y_test)
                
                st.session_state.current_df = df
                st.session_state.label_encoders = label_encoders
                st.session_state.models = models
                st.session_state.scores_df = scores_df
                st.success("Dataset updated successfully")
            except Exception as e:
                st.error(f"Error updating dataset: {str(e)}")
    
    with st.expander("System Info"):
        st.metric("Dataset Rows", len(st.session_state.current_df))
        st.metric("Features", len(st.session_state.X.columns))
        st.metric("Models", len(st.session_state.models))

# --- Sidebar Navigation ---
def create_sidebar():
    st.sidebar.title("Navigation")
    admin_mode = st.sidebar.checkbox("Admin Mode")
    
    pages = ["Home", "Data Analysis", "Prediction", "Reports", "Help"]
    if admin_mode:
        pages.append("Admin")
    
    for page in pages:
        if st.sidebar.button(page, key=f"nav_{page}"):
            navigate_to(page)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Dataset:** {len(st.session_state.current_df)} rows")
    st.sidebar.markdown(f"**Target:** {st.session_state.target_col}")

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
