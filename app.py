# Streamlit App: app.py

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
from collections import Counter
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Handle SMOTE import with fallback
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    from imblearn.over_sampling import RandomOverSampler as SMOTE

# Set dark theme for visualizations
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

# Configure Streamlit page
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
    .card {
        background-color: #1E2130; border-radius: 8px; padding: 15px; 
        margin-bottom: 15px; border: 1px solid #2A3459;
    }
    .card-title {
        font-size: 1.2em; font-weight: bold; margin-bottom: 10px; color: #4A8DF8;
    }
</style>
""", unsafe_allow_html=True)

# Project description
PROJECT_OVERVIEW = """
<div class="card">
    <div class="card-title">Project Overview</div>
    <p>Traffic accidents are a major problem worldwide, causing several fatalities, damage to property, and loss of productivity. Predicting accident severity based on contributors such as weather conditions, vehicle damage extent, and drivers enables the authorities to take necessary actions to minimize the risk and develop better emergency responses.</p>
    <p>This project uses machine learning techniques to analyze past traffic data for accident severity prediction and present useful data to improve road safety and management.</p>
</div>
"""

# Initialize session state
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
if 'is_default_data' not in st.session_state:
    st.session_state.is_default_data = True

# Navigation function
def navigate_to(page):
    st.session_state.current_page = page

# Load and cache default data
@st.cache_data(persist="disk")
def load_default_data():
    url = st.session_state.default_dataset
    df = pd.read_csv(url)
    return preprocess_data(df, is_default=True)

# Preprocess data function
def preprocess_data(df, is_default=False):
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    
    if 'Location' in df.columns:
        try:
            location = df['Location'].str.replace(r'[()]', '', regex=True).str.split(', ', expand=True)
            df['latitude'] = location[0].astype(float)
            df['longitude'] = location[1].astype(float)
        except:
            pass
    
    target_col = st.session_state.target_col
    if target_col not in df.columns:
        possible_targets = [col for col in df.columns if 'severity' in col.lower() or 'injury' in col.lower()]
        if possible_targets:
            target_col = possible_targets[0]
            st.session_state.target_col = target_col
    
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col != 'Location':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    numeric_cols = df.select_dtypes(include='number').columns.difference([target_col])
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    st.session_state.is_default_data = is_default
    return df, label_encoders, target_col

# Prepare data for modeling
def prepare_model_data(df, target_col):
    X = df.drop([target_col, 'Location'], axis=1, errors='ignore')
    y = df[target_col].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    if st.session_state.is_default_data:
        try:
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            return X, y, X_train_resampled, X_test, y_train_resampled, y_test
        except Exception:
            return X, y, X_train, X_test, y_train, y_test
    
    return X, y, X_train, X_test, y_train, y_test

# Train and cache models
@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'Artificial Neural Network': MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=42
        )
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

# Load default data if none exists
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

# Home page
def render_home():
    st.title("Traffic Accident Severity Prediction")
    st.markdown(PROJECT_OVERVIEW, unsafe_allow_html=True)

    if st.session_state.current_df is not None:
        with st.expander("Dataset Preview", expanded=True):
            st.dataframe(st.session_state.current_df.head().style.set_properties(**{
                'background-color': '#1E2130',
                'color': 'white',
                'border-color': '#2A3459'
            }))
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(st.session_state.current_df))
        col2.metric("Features Available", len(st.session_state.current_df.columns))
        col3.metric("Trained Models", len(st.session_state.models))
    else:
        st.error("No data loaded. Please check the dataset.")

# Data analysis page
def render_data_analysis():
    st.title("Data Analysis & Insights")

    if st.session_state.current_df is None:
        st.error("No data loaded. Please check the dataset.")
        return

    df = st.session_state.current_df
    scores_df = st.session_state.scores_df

    with st.expander("Target Variable Distribution", expanded=True):
        if st.session_state.target_col in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x=st.session_state.target_col, data=df, ax=ax, palette="coolwarm")
            
            ax.set_title(f'Count of {st.session_state.target_col} Levels', color='white')
            ax.set_xlabel('Severity Level', color='white')
            ax.set_ylabel('Count', color='white')
            st.pyplot(fig)
        else:
            st.warning(f"Target column '{st.session_state.target_col}' not found in dataset.")

    with st.expander("Accident Hotspot Locations"):
        if 'latitude' in df.columns and 'longitude' in df.columns:
            m = folium.Map(
                location=[df['latitude'].mean(), df['longitude'].mean()], 
                zoom_start=11, 
                tiles='CartoDB dark_matter',
                attr='© OpenStreetMap contributors © CARTO'
            )

            for idx, row in df.sample(min(1000, len(df))).iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3, color='red', fill=True, fill_color='red', fill_opacity=0.9
                ).add_to(m)

            folium_static(m, width=1000, height=600)
        else:
            st.warning("No location data found in dataset.")

    with st.expander("Feature Correlation Heatmap"):
        try:
            corr = df.select_dtypes(['number']).corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax, center=0,
                       cbar_kws={'label': 'Correlation Coefficient'})
            ax.set_title("Feature Correlation Heatmap", color='white', pad=20)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate correlation heatmap: {str(e)}")

    if not st.session_state.scores_df.empty:
        with st.expander("Model Performance Metrics"):
            formatted_scores = st.session_state.scores_df.copy()
            for col in formatted_scores.columns[1:]:
                formatted_scores[col] = formatted_scores[col].apply(lambda x: f"{x:.2f}")

            st.table(formatted_scores.style.set_properties(**{
                'background-color': '#1E2130',
                'color': 'white',
                'border-color': '#2A3459'
            }))
            
            st.subheader("Model Performance Comparison")
            performance_df = st.session_state.scores_df.set_index('Model')
            fig, ax = plt.subplots(figsize=(10, 6))
            performance_df.plot(kind='bar', ax=ax, cmap='coolwarm')
            ax.set_title('Model Performance Comparison', color='white', pad=20)
            ax.set_ylabel('Score (%)', color='white')
            ax.set_xlabel('Model', color='white')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(facecolor='#0E1117', edgecolor='#0E1117')
            st.pyplot(fig)
    
    with st.expander("Feature Importance Analysis"):
        if st.session_state.models:
            model_name = st.selectbox("Select Model", list(st.session_state.models.keys()), index=1)
            model = st.session_state.models[model_name]

            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_[0])
                elif hasattr(model, 'coefs_'):
                    importances = np.mean(np.abs(model.coefs_[0]), axis=1)
                else:
                    raise AttributeError("Model doesn't have feature importance attributes")

                importances_vals = importances / importances.sum()
                sorted_idx = np.argsort(importances_vals)[::-1]
                
                n_features = min(10, len(st.session_state.X.columns))
                top_features = st.session_state.X.columns[sorted_idx][:n_features]
                top_vals = importances_vals[sorted_idx][:n_features]

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=top_vals, y=top_features, ax=ax, palette="coolwarm")
                ax.set_title(f'{model_name} - Top {n_features} Features', color='white', pad=20)
                ax.set_xlabel('Importance Score', color='white')
                ax.set_ylabel('Feature', color='white')
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not display feature importances: {str(e)}")
        else:
            st.warning("No trained models available.")

# Prediction page
def render_prediction():
    st.title("Accident Severity Prediction")
    st.markdown("Make custom predictions by selecting input values below.")

    if not st.session_state.models:
        st.warning("No models available for prediction. Please check the Data Analysis page.")
        return

    with st.container():
        st.subheader("Model Selection")
        selected_model = st.selectbox("Select Prediction Model", list(st.session_state.models.keys()))
        model = st.session_state.models[selected_model]

    with st.container():
        st.subheader("Input Parameters")
        col1, col2 = st.columns(2)

        input_data = {}
        for i, col in enumerate(st.session_state.X.columns):
            current_col = col1 if i % 2 == 0 else col2

            if col in st.session_state.label_encoders:
                options = sorted(st.session_state.label_encoders[col].classes_)
                choice = current_col.selectbox(f"{col}", options)
                input_data[col] = st.session_state.label_encoders[col].transform([choice])[0]
            else:
                col_min = st.session_state.current_df[col].min()
                col_max = st.session_state.current_df[col].max()
                col_mean = st.session_state.current_df[col].mean()
                input_data[col] = current_col.number_input(
                    f"{col}", float(col_min), float(col_max), float(col_mean), key=f"input_{col}"
                )

        if st.button("Predict Severity", key="predict_button"):
            input_df = pd.DataFrame([input_data])
            try:
                prediction = model.predict(input_df)[0]
                probs = model.predict_proba(input_df)[0]
                confidence = np.max(probs) * 100

                if st.session_state.target_col in st.session_state.label_encoders:
                    severity_label = st.session_state.label_encoders[st.session_state.target_col].inverse_transform([prediction])[0]
                else:
                    severity_label = prediction

                with st.container():
                    st.subheader("Prediction Results")
                    res_col1, res_col2 = st.columns(2)

                    res_col1.markdown(f"""
                    <div class="card">
                        <div class="card-title">Predicted Severity</div>
                        <h2 style="color: #4A8DF8;">{severity_label}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                    res_col2.markdown(f"""
                    <div class="card">
                        <div class="card-title">Confidence Level</div>
                        <h2 style="color: #4A8DF8;">{confidence:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# Reports page
def render_reports():
    st.title("Dataset Reports")

    if st.session_state.current_df is None:
        st.error("No data loaded. Please check the dataset.")
        return

    with st.expander("Dataset Summary Statistics", expanded=True):
        st.markdown("""
        <div class="card">
            <div class="card-title">Statistical Overview</div>
            <p>Basic statistics for numerical columns in the dataset.</p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(st.session_state.current_df.describe().style.set_properties(**{
            'background-color': '#1E2130',
            'color': 'white',
            'border-color': '#2A3459'
        }))

    with st.expander("Column Information"):
        st.markdown("""
        <div class="card">
            <div class="card-title">Column Details</div>
            <p>Information about each column in the dataset.</p>
        </div>
        """, unsafe_allow_html=True)
        col_info = pd.DataFrame({
            'Column': st.session_state.current_df.columns,
            'Data Type': st.session_state.current_df.dtypes,
            'Unique Values': [st.session_state.current_df[col].nunique() for col in st.session_state.current_df.columns]
        })
        st.dataframe(col_info.style.set_properties(**{
            'background-color': '#1E2130',
            'color': 'white',
            'border-color': '#2A3459'
        }))

# Help page
def render_help():
    st.title("User Guide")

    with st.expander("Application Overview", expanded=True):
        st.markdown("""
        <div class="card">
            <div class="card-title">Application Overview</div>
            <p>This application provides tools for analyzing traffic accident data and predicting accident severity using machine learning models.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Navigation Guide"):
        st.markdown("""
        <div class="card">
            <div class="card-title">Navigation Guide</div>
            <p>Use the sidebar to navigate between different sections:</p>
            <ul>
                <li><b>Home</b>: Project overview and dataset preview</li>
                <li><b>Data Analysis</b>: Visualizations and insights from the data</li>
                <li><b>Prediction</b>: Make custom severity predictions</li>
                <li><b>Reports</b>: View detailed dataset information</li>
                <li><b>Help</b>: This user guide</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("Data Analysis Section"):
        st.markdown("""
        <div class="card">
            <div class="card-title">Data Analysis Section</div>
            <p>The Data Analysis page includes:</p>
            <ul>
                <li>Target variable distribution</li>
                <li>Accident location hotspots</li>
                <li>Feature correlation analysis</li>
                <li>Model performance metrics</li>
                <li>Feature importance analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("Prediction Section"):
        st.markdown("""
        <div class="card">
            <div class="card-title">Prediction Section</div>
            <p>To make predictions:</p>
            <ol>
                <li>Select a machine learning model</li>
                <li>Set the input parameters</li>
                <li>Click "Predict Severity"</li>
                <li>View the results</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("Technical Information"):
        st.markdown("""
        <div class="card">
            <div class="card-title">Technical Information</div>
            <p>The application uses the following machine learning models:</p>
            <ul>
                <li>Logistic Regression</li>
                <li>Random Forest</li>
                <li>XGBoost</li>
                <li>Artificial Neural Network</li>
            </ul>
            <p>All visualizations use a consistent dark theme for better readability.</p>
        </div>
        """, unsafe_allow_html=True)

# Admin page
def render_admin():
    st.title("Administration Dashboard")
    password = st.text_input("Enter Admin Password:", type="password", key="admin_password")

    if password != "admin1":
        st.error("Incorrect password. Access denied.")
        st.stop()

    st.warning("You are in administrator mode. Changes here will affect all users.")

    with st.expander("Dataset Management", expanded=True):
        st.markdown("""
        <div class="card">
            <div class="card-title">Upload New Dataset</div>
            <p>Upload a new CSV file to update the system dataset.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Select CSV file", type="csv", key="dataset_uploader")

        if uploaded_file is not None:
            st.info("File uploaded successfully. Click the button below to update the system.")
            if st.button("Update System with New Dataset", key="update_dataset"):
                with st.spinner("Processing new dataset and retraining models..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        new_df = pd.read_csv(tmp_path)
                        os.unlink(tmp_path)
                        
                        new_df, new_label_encoders, new_target_col = preprocess_data(new_df, is_default=False)
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
                        
                        st.success("Dataset updated successfully! All pages have been refreshed with the new data.")
                    except Exception as e:
                        st.error(f"Error processing uploaded file: {str(e)}")

    with st.expander("System Information"):
        st.subheader("System Information")
        info_col1, info_col2 = st.columns(2)

        info_col1.markdown(f"""
        <div class="card">
            <div class="card-title">Current Target Variable</div>
            <h3>{st.session_state.target_col}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        info_col1.markdown(f"""
        <div class="card">
            <div class="card-title">Number of Features</div>
            <h3>{len(st.session_state.X.columns) if st.session_state.X is not None else 0}</h3>
        </div>
        """, unsafe_allow_html=True)

        info_col2.markdown(f"""
        <div class="card">
            <div class="card-title">Number of Models</div>
            <h3>{len(st.session_state.models)}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        info_col2.markdown(f"""
        <div class="card">
            <div class="card-title">Dataset Rows</div>
            <h3>{len(st.session_state.current_df)}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("System Maintenance"):
        st.markdown("""
        <div class="card">
            <div class="card-title">Reset System</div>
            <p>Reset to the default dataset configuration.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Reset to Default Dataset", key="reset_system"):
            with st.spinner("Resetting to default dataset..."):
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
                    st.session_state.target_col = target_col
                    st.session_state.is_default_data = True

                    st.success("System reset to default dataset completed!")
                except Exception as e:
                    st.error(f"Reset failed: {str(e)}")

# Create sidebar navigation
def create_sidebar():
    st.sidebar.title("Navigation")
    
    admin_mode = st.sidebar.checkbox("Admin Mode", key="admin_mode")
    
    pages = ["Home", "Data Analysis", "Prediction", "Reports", "Help"]
    if admin_mode:
        pages.append("Admin")
    
    for page in pages:
        if st.sidebar.button(page, key=f"nav_{page}"):
            navigate_to(page)

# Main app function
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
