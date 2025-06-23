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

# --- Custom Dark Theme with Red Accents ---
def set_custom_theme():
    # Custom color palette with red accents
    PALETTE = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', 
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Set seaborn style
    sns.set_style("darkgrid", {
        "axes.facecolor": "#0E1117",
        "axes.edgecolor": "#2A3459",
        "axes.labelcolor": "white",
        "figure.facecolor": "#0E1117",
        "grid.color": "#2A3459",
        "text.color": "white",
        "xtick.color": "white",
        "ytick.color": "white"
    })
    
    # Set matplotlib rcParams
    plt.rcParams['figure.facecolor'] = '#0E1117'
    plt.rcParams['axes.facecolor'] = '#0E1117'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['grid.color'] = '#2A3459'
    
    return PALETTE

PALETTE = set_custom_theme()

# --- Streamlit Config ---
st.set_page_config(page_title="Accident Severity Predictor", layout="wide")

# Custom CSS for dark theme with red accents
st.markdown(f"""
<style>
    /* Main page background */
    .stApp {{
        background-color: #0E1117;
        color: white;
    }}
    
    /* Sidebar background */
    .css-1d391kg {{
        background-color: #0E1117;
        border-right: 1px solid #2A3459;
    }}
    
    /* Radio button styling */
    .st-bb, .st-bc, .st-bd, .st-be, .st-bf, .st-bg {{
        background-color: #1E2130;
    }}
    
    /* Selected radio button */
    .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak, .st-al, .st-am, .st-an, .st-ao, .st-ap, .st-aq, .st-ar, .st-as {{
        background-color: #d62728;
        color: white;
    }}
    
    /* Widgets */
    .stTextInput input, .stSelectbox select, .stNumberInput input {{
        color: white !important;
    }}
    
    /* Dataframes and tables */
    .stDataFrame, table {{
        background-color: #1E2130;
        color: white !important;
    }}
    
    /* Markdown text color */
    .stMarkdown {{
        color: white;
    }}
    
    /* Divider color */
    hr {{
        border-color: #2A3459;
    }}
    
    /* Button styling */
    .stButton>button {{
        background-color: #1E2130;
        color: white;
        border-color: #2A3459;
    }}
    
    .stButton>button:hover {{
        background-color: #d62728;
        color: white;
    }}
    
    /* Success, info, warning, error boxes */
    .stAlert {{
        background-color: #1E2130;
        border-color: #2A3459;
    }}
    
    /* Custom header styling */
    .header {{
        color: #d62728;
        border-bottom: 2px solid #d62728;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }}
    
    /* Card styling */
    .card {{
        background-color: #1E2130;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 4px solid #d62728;
    }}
</style>
""", unsafe_allow_html=True)

# --- Project Overview ---
PROJECT_OVERVIEW = """
Traffic accidents are a major problem worldwide, causing several fatalities, damage to property, and loss of productivity. Predicting accident severity based on contributors such as weather conditions, road conditions, types of vehicles, and drivers enables the authorities to take necessary actions to minimize the risk and develop better emergency responses. 
 
This project uses machine learning techniques to analyze past traffic data for accident severity prediction and present useful data to improve road safety and management.
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

# [Previous functions remain unchanged... load_default_data, preprocess_data, prepare_model_data, train_models, etc.]

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

# --- Navigation Panel ---
st.sidebar.title("Accident Severity Predictor")
st.sidebar.markdown("---")

# Custom radio button navigation
nav_options = ["Home", "Data Analysis", "Prediction", "Reports", "Help"]
if st.sidebar.checkbox("Admin Mode", key="admin_mode"):
    nav_options.append("Admin")

# Create styled radio buttons
page = st.sidebar.radio(
    "Navigation",
    nav_options,
    format_func=lambda x: f"📌 {x}" if x == "Home" else f"📊 {x}" if x == "Data Analysis" else f"🔮 {x}" if x == "Prediction" else f"📋 {x}" if x == "Reports" else f"❓ {x}" if x == "Help" else f"🔑 {x}",
    key="nav_radio"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("This app predicts accident severity using machine learning models.")

# --- Home Page ---
if page == "Home":
    st.markdown('<div class="header"><h1>Traffic Accident Severity Prediction</h1></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown(PROJECT_OVERVIEW)
        
        st.markdown("---")
        st.markdown('<div class="card"><h3>Dataset Preview</h3></div>', unsafe_allow_html=True)
        st.dataframe(st.session_state.current_df.copy().head(), use_container_width=True)

# --- Data Analysis Page ---
elif page == "Data Analysis":
    st.markdown('<div class="header"><h1>Data Analysis & Insights</h1></div>', unsafe_allow_html=True)
    st.markdown("Explore key patterns and model performance metrics.")
    
    df = st.session_state.current_df
    scores_df = st.session_state.scores_df
    
    with st.container():
        st.markdown('<div class="card"><h3>Target Variable Distribution</h3></div>', unsafe_allow_html=True)
        if st.session_state.target_col in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x=st.session_state.target_col, data=df, ax=ax, palette=PALETTE)
            
            severity_labels = {
                0: "No Injury",
                1: "Minor Injury",
                2: "Moderate Injury",
                3: "Serious Injury",
                4: "Fatal Injury"
            }
            
            current_labels = [int(tick.get_text()) for tick in ax.get_xticklabels()]
            new_labels = [severity_labels.get(label, label) for label in current_labels]
            ax.set_xticklabels(new_labels, rotation=45, ha='right')
            
            ax.set_title(f'Count of {st.session_state.target_col} Levels', color='white', pad=20)
            ax.set_xlabel('Severity Level', color='white')
            ax.set_ylabel('Count', color='white')
            st.pyplot(fig)
        else:
            st.warning(f"Target column '{st.session_state.target_col}' not found in dataset.")
    
    with st.container():
        st.markdown('<div class="card"><h3>Accident Hotspot Locations</h3></div>', unsafe_allow_html=True)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], 
                          zoom_start=11, 
                          tiles='CartoDB dark_matter')
            
            for idx, row in df.sample(min(1000, len(df))).iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    color=PALETTE[0],  # Using the red color from our palette
                    fill=True,
                    fill_color=PALETTE[0],
                    fill_opacity=0.7
                ).add_to(m)
            
            folium_static(m, width=1000, height=500)
        else:
            st.warning("No location data found in dataset.")
    
    with st.container():
        st.markdown('<div class="card"><h3>Feature Correlation Analysis</h3></div>', unsafe_allow_html=True)
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
        with st.container():
            st.markdown('<div class="card"><h3>Model Performance Metrics</h3></div>', unsafe_allow_html=True)
            formatted_scores = st.session_state.scores_df.copy()
            for col in formatted_scores.columns[1:]:
                formatted_scores[col] = formatted_scores[col].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(formatted_scores.style.applymap(lambda x: 'color: #d62728' if '%' in x else ''), use_container_width=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            performance_df = st.session_state.scores_df.set_index('Model')
            performance_df.plot(kind='bar', ax=ax, color=PALETTE)
            ax.set_title('Model Performance Comparison', color='white', pad=20)
            ax.set_ylabel('Score (%)', color='white')
            ax.set_xlabel('Model', color='white')
            ax.legend(facecolor='#0E1117', edgecolor='#0E1117')
            st.pyplot(fig)
        
        with st.container():
            st.markdown('<div class="card"><h3>Feature Importance Analysis</h3></div>', unsafe_allow_html=True)
            model_name = st.selectbox("Select Model", list(st.session_state.models.keys()), index=1)
            
            if model_name in st.session_state.models:
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
                    sns.barplot(x=top_vals, y=top_features, ax=ax, palette=PALETTE)
                    ax.set_title(f'{model_name} - Top {n_features} Features', color='white', pad=20)
                    ax.set_xlabel('Importance Score', color='white')
                    ax.set_ylabel('Feature', color='white')
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not display feature importances: {str(e)}")

# --- Prediction Page ---
elif page == "Prediction":
    st.markdown('<div class="header"><h1>Accident Severity Prediction</h1></div>', unsafe_allow_html=True)
    st.markdown("Make custom predictions by selecting input values below.")
    
    if not st.session_state.models:
        st.warning("No models available for prediction. Please check the Data Analysis page.")
    else:
        with st.container():
            st.markdown('<div class="card"><h3>Model Selection</h3></div>', unsafe_allow_html=True)
            selected_model = st.selectbox("Select Prediction Model", list(st.session_state.models.keys()))
            model = st.session_state.models[selected_model]
        
        with st.container():
            st.markdown('<div class="card"><h3>Input Parameters</h3></div>', unsafe_allow_html=True)
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
                        f"{col}", 
                        float(col_min), 
                        float(col_max), 
                        float(col_mean),
                        key=f"input_{col}"
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
                        st.markdown('<div class="card"><h3>Prediction Results</h3></div>', unsafe_allow_html=True)
                        res_col1, res_col2 = st.columns(2)
                        
                        with res_col1:
                            st.metric(label="Predicted Severity", value=severity_label)
                        
                        with res_col2:
                            st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
                        
                        if st.session_state.target_col in st.session_state.label_encoders:
                            st.markdown('<div class="card"><h4>Probability Distribution</h4></div>', unsafe_allow_html=True)
                            prob_df = pd.DataFrame({
                                'Severity Level': st.session_state.label_encoders[st.session_state.target_col].classes_,
                                'Probability': probs * 100
                            })
                            
                            fig, ax = plt.subplots(figsize=(10, 4))
                            sns.barplot(x='Severity Level', y='Probability', data=prob_df, 
                                        ax=ax, palette=PALETTE)
                            ax.set_title('Severity Probability Distribution', color='white')
                            ax.set_xlabel('Severity Level', color='white')
                            ax.set_ylabel('Probability (%)', color='white')
                            st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

# --- Reports Page ---
elif page == "Reports":
    st.markdown('<div class="header"><h1>Dataset Reports</h1></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card"><h3>Dataset Summary Statistics</h3></div>', unsafe_allow_html=True)
        st.dataframe(st.session_state.current_df.describe(), use_container_width=True)
    
    with st.container():
        st.markdown('<div class="card"><h3>Column Information</h3></div>', unsafe_allow_html=True)
        col_info = pd.DataFrame({
            'Column': st.session_state.current_df.columns,
            'Data Type': st.session_state.current_df.dtypes,
            'Unique Values': [st.session_state.current_df[col].nunique() for col in st.session_state.current_df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
    
    with st.container():
        st.markdown('<div class="card"><h3>Missing Values Report</h3></div>', unsafe_allow_html=True)
        missing_data = st.session_state.current_df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            st.write("Columns with missing values:")
            st.dataframe(missing_data.reset_index().rename(columns={'index': 'Column', 0: 'Missing Values'}), use_container_width=True)
        else:
            st.success("No missing values found in the dataset.")

# --- Help Page ---
elif page == "Help":
    st.markdown('<div class="header"><h1>User Guide</h1></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card"><h3>Application Overview</h3></div>', unsafe_allow_html=True)
        st.markdown("""
        This application provides tools for analyzing traffic accident data and predicting accident severity using machine learning models.
        """)
    
    with st.container():
        st.markdown('<div class="card"><h3>Navigation Guide</h3></div>', unsafe_allow_html=True)
        st.markdown("""
        Use the sidebar to navigate between different sections:
        
        - **Home**: Project overview and dataset preview
        - **Data Analysis**: Visualizations and insights from the data
        - **Prediction**: Make custom severity predictions
        - **Reports**: View detailed dataset information
        - **Help**: This user guide
        """)
    
    with st.container():
        st.markdown('<div class="card"><h3>Data Analysis Section</h3></div>', unsafe_allow_html=True)
        st.markdown("""
        The Data Analysis page includes:
        - Target variable distribution
        - Accident location hotspots
        - Feature correlation analysis
        - Model performance metrics
        - Feature importance analysis
        """)
    
    with st.container():
        st.markdown('<div class="card"><h3>Prediction Section</h3></div>', unsafe_allow_html=True)
        st.markdown("""
        To make predictions:
        1. Select a machine learning model
        2. Set the input parameters
        3. Click "Predict Severity"
        4. View the results
        """)
    
    with st.container():
        st.markdown('<div class="card"><h3>Technical Information</h3></div>', unsafe_allow_html=True)
        st.markdown("""
        The application uses the following machine learning models:
        - Logistic Regression
        - Random Forest
        - XGBoost
        - Artificial Neural Network
        
        All visualizations use a consistent dark theme with red accents for better readability.
        """)

# --- Admin Page ---
elif page == "Admin":
    st.markdown('<div class="header"><h1>Administration Dashboard</h1></div>', unsafe_allow_html=True)
    
    # Password protection
    password = st.text_input("Enter Admin Password:", type="password", key="admin_password")
    
    if password != "admin1":
        st.error("Incorrect password. Access denied.")
        st.stop()
    
    st.warning("You are in administrator mode. Changes here will affect all users.")
    
    with st.container():
        st.markdown('<div class="card"><h3>Dataset Management</h3></div>', unsafe_allow_html=True)
        with st.expander("Upload New Dataset"):
            uploaded_file = st.file_uploader("Select CSV file", type="csv", key="dataset_uploader")
            
            if uploaded_file is not None:
                st.info("File uploaded successfully. Click the button below to update the system.")
                if st.button("Update System with New Dataset", key="update_dataset"):
                    with st.spinner("Processing new dataset and retraining models..."):
                        handle_dataset_upload(uploaded_file)
    
    with st.container():
        st.markdown('<div class="card"><h3>System Information</h3></div>', unsafe_allow_html=True)
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.metric("Current Target Variable", st.session_state.target_col)
            st.metric("Number of Features", len(st.session_state.X.columns) if st.session_state.X is not None else 0)
        
        with info_col2:
            st.metric("Number of Models", len(st.session_state.models))
            st.metric("Dataset Rows", len(st.session_state.current_df))
    
    with st.container():
        st.markdown('<div class="card"><h3>System Maintenance</h3></div>', unsafe_allow_html=True)
        if st.button("Reset to Default Dataset", key="reset_system"):
            with st.spinner("Resetting to default dataset..."):
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
                
                st.success("System reset to default dataset completed!")
