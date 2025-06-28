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
    .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak, .st-al, .st-am, .st-an, .st-ao, .st-ap, .st-aq, .st-ar, .st-as {
        background-color: #1E2130; color: white; border-color: #2A3459;
    }
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
    # Data cleaning
    df = df.copy()
    df.drop_duplicates(inplace=True)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    for col in df.select_dtypes(exclude=np.number).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Feature engineering - label encoding
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col != 'Location':  # Skip location column for encoding
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Extract coordinates from Location
    if 'Location' in df.columns:
        location = df['Location'].str.replace(r'[()]', '', regex=True).str.split(',', expand=True)
        df['latitude'] = location[0].astype(float)
        df['longitude'] = location[1].astype(float)
    
    # Normalize numeric columns
    target_col = 'Injury Severity'
    numeric_cols = df.select_dtypes(include=np.number).columns.difference([target_col])
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, label_encoders, target_col

def prepare_model_data(df, target_col):
    X = df.drop([target_col, 'Location'], axis=1, errors='ignore')
    y = df[target_col]
    
    # Convert y to numpy array if it's a pandas Series
    if hasattr(y, 'values'):
        y = y.values
    
    # Train-test split before SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Apply SMOTE only to training data
    smote = SMOTE(random_state=42)
    try:
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        return X, y, X_train_resampled, X_test, y_train_resampled, y_test
    except ValueError as e:
        st.error(f"SMOTE error: {str(e)}")
        return X, y, X_train, X_test, y_train, y_test

# --- Model Training ---
@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'Artificial Neural Network': MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=300, 
            activation='relu', solver='adam', random_state=42
        )
    }
    
    trained_models = {}
    model_scores = []

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            trained_models[name] = model
            model_scores.append([name, acc*100, prec*100, rec*100, f1*100])
        except Exception as e:
            st.warning(f"Failed to train {name}: {str(e)}")
            continue

    scores_df = pd.DataFrame(model_scores, columns=['Model', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'])
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
        st.error(f"Initialization error: {str(e)}")

# --- Admin Page Functions ---
def handle_dataset_upload(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        new_df = pd.read_csv(tmp_path)
        os.unlink(tmp_path)

        new_df, new_label_encoders, new_target_col = preprocess_data(new_df)
        new_X, new_y, new_X_train, new_X_test, new_y_train, new_y_test = prepare_model_data(new_df, new_target_col)
        new_models, new_scores_df = train_models(new_X_train, new_y_train, new_X_test, new_y_test)

        st.session_state.current_df = new_df
        st.session_state.label_encoders = new_label_encoders
        st.session_state.models = new_models
        st.session_state.scores_df = new_scores_df
        st.session_state.X = new_X
        st.session_state.y = new_y
        st.session_state.X_train = new_X_train
        st.session_state.X_test = new_X_test
        st.session_state.y_train = new_y_train
        st.session_state.y_test = new_y_test
        st.session_state.target_col = new_target_col

        st.success("Dataset updated successfully!")
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")

# --- Page Rendering Functions ---
def render_home():
    st.title("Traffic Accident Severity Prediction")
    st.markdown(PROJECT_OVERVIEW, unsafe_allow_html=True)

    with st.expander("Dataset Preview", expanded=True):
        st.dataframe(st.session_state.current_df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(st.session_state.current_df))
    col2.metric("Features", len(st.session_state.current_df.columns))
    col3.metric("Models", len(st.session_state.models))

def render_data_analysis():
    st.title("Data Analysis & Insights")
    
    df = st.session_state.current_df
    scores_df = st.session_state.scores_df

    with st.expander("Target Variable Distribution", expanded=True):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=st.session_state.target_col, data=df, ax=ax, palette="coolwarm")
        ax.set_title(f'Distribution of {st.session_state.target_col}', color='white')
        ax.set_xlabel('Severity Level', color='white')
        ax.set_ylabel('Count', color='white')
        st.pyplot(fig)

    if 'latitude' in df.columns and 'longitude' in df.columns:
        with st.expander("Accident Hotspot Locations"):
            m = folium.Map(
                location=[df['latitude'].mean(), df['longitude'].mean()],
                zoom_start=11,
                tiles='CartoDB dark_matter'
            )
            sample_df = df.sample(min(1000, len(df)))
            for _, row in sample_df.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.7
                ).add_to(m)
            folium_static(m, width=1000, height=600)

    with st.expander("Feature Correlation Heatmap"):
        try:
            corr = df.select_dtypes(include=np.number).corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax, center=0)
            ax.set_title("Feature Correlation Heatmap", color='white')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate heatmap: {str(e)}")

    if not scores_df.empty:
        with st.expander("Model Performance"):
            st.table(scores_df.style.format({
                'Accuracy (%)': '{:.2f}',
                'Precision (%)': '{:.2f}',
                'Recall (%)': '{:.2f}',
                'F1-Score (%)': '{:.2f}'
            }))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scores_df.set_index('Model').plot(kind='bar', ax=ax, cmap='coolwarm')
            ax.set_title('Model Performance Comparison', color='white')
            ax.set_ylabel('Score (%)', color='white')
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

    with st.expander("Feature Importance"):
        model_name = st.selectbox("Select Model", list(st.session_state.models.keys()))
        model = st.session_state.models[model_name]

        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            elif hasattr(model, 'coefs_'):
                importances = np.mean(np.abs(model.coefs_[0]), axis=1)
            else:
                raise AttributeError("No feature importance method")

            importance_df = pd.DataFrame({
                'Feature': st.session_state.X.columns,
                'Importance': importances / importances.sum()
            }).sort_values('Importance', ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, palette="coolwarm")
            ax.set_title(f'{model_name} Feature Importance', color='white')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not display feature importances: {str(e)}")

def render_prediction():
    st.title("Accident Severity Prediction")
    
    if not st.session_state.models:
        st.warning("No trained models available")
        return

    selected_model = st.selectbox("Select Model", list(st.session_state.models.keys()))
    model = st.session_state.models[selected_model]

    input_data = {}
    cols = st.columns(2)
    for i, col in enumerate(st.session_state.X.columns):
        current_col = cols[i % 2]
        if col in st.session_state.label_encoders:
            options = st.session_state.label_encoders[col].classes_
            choice = current_col.selectbox(col, options)
            input_data[col] = st.session_state.label_encoders[col].transform([choice])[0]
        else:
            col_data = st.session_state.current_df[col]
            val = current_col.number_input(
                col,
                float(col_data.min()),
                float(col_data.max()),
                float(col_data.median())
            )
            input_data[col] = val

    if st.button("Predict Severity"):
        try:
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            probs = model.predict_proba(input_df)[0]
            confidence = np.max(probs) * 100

            if st.session_state.target_col in st.session_state.label_encoders:
                severity_label = st.session_state.label_encoders[st.session_state.target_col].inverse_transform([prediction])[0]
            else:
                severity_label = prediction

            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            col1.metric("Predicted Severity", severity_label)
            col2.metric("Confidence", f"{confidence:.2f}%")

            if st.session_state.target_col in st.session_state.label_encoders:
                st.subheader("Probability Distribution")
                prob_df = pd.DataFrame({
                    'Severity Level': st.session_state.label_encoders[st.session_state.target_col].classes_,
                    'Probability': probs * 100
                })
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(x='Severity Level', y='Probability', data=prob_df, palette="coolwarm")
                ax.set_title('Probability Distribution', color='white')
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# --- Other page functions (Reports, Help, Admin) remain the same ---
# [Previous implementations of render_reports(), render_help(), render_admin() can be inserted here]

# --- Sidebar and Main App ---
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
