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

# --- Ocean Color Scheme Configuration ---
def set_ocean_theme():
    # Set seaborn style
    sns.set_style("darkgrid")
    
    # Custom color palette (blues and teals)
    PALETTE = ['#1a6985', '#3a9bc8', '#5fcde4', '#83e2d6', '#a6f7d1',
               '#d4f0fc', '#8bb8e8', '#4a7bb7', '#2e4e7e', '#1d2d50']
    
    # Set matplotlib rcParams
    plt.rcParams['figure.facecolor'] = '#0c1445'
    plt.rcParams['axes.facecolor'] = '#0c1445'
    plt.rcParams['axes.edgecolor'] = '#5fcde4'
    plt.rcParams['axes.labelcolor'] = '#d4f0fc'
    plt.rcParams['text.color'] = '#d4f0fc'
    plt.rcParams['xtick.color'] = '#a6f7d1'
    plt.rcParams['ytick.color'] = '#a6f7d1'
    plt.rcParams['grid.color'] = '#1d2d50'
    
    return PALETTE

PALETTE = set_ocean_theme()

# --- Streamlit Config ---
st.set_page_config(page_title="Accident Severity Predictor", layout="wide", page_icon="🚗")

# Custom CSS for ocean theme
st.markdown("""
<style>
    /* Main page background */
    .stApp {
        background-color: #0c1445;
        color: #d4f0fc;
    }
    
    /* Sidebar background */
    .css-1d391kg {
        background-color: #0c1445;
        border-right: 1px solid #1d2d50;
    }
    
    /* Widgets */
    .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak, .st-al, .st-am, .st-an, .st-ao, .st-ap, .st-aq, .st-ar, .st-as {
        background-color: #1d2d50;
        color: #d4f0fc;
        border-color: #3a9bc8;
    }
    
    /* Text input */
    .stTextInput input {
        color: #d4f0fc !important;
    }
    
    /* Select boxes */
    .stSelectbox select {
        color: #d4f0fc !important;
    }
    
    /* Number input */
    .stNumberInput input {
        color: #d4f0fc !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #1d2d50;
    }
    
    /* Tables */
    table {
        color: #d4f0fc !important;
    }
    
    /* Markdown text color */
    .stMarkdown {
        color: #d4f0fc;
    }
    
    /* Divider color */
    hr {
        border-color: #3a9bc8;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #1d2d50;
        color: #d4f0fc;
        border-color: #3a9bc8;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #3a9bc8;
        color: #0c1445;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(58, 155, 200, 0.3);
    }
    
    /* Navigation buttons */
    .nav-btn {
        width: 100%;
        margin: 5px 0;
        padding: 10px;
        border-radius: 8px;
        background-color: #1d2d50;
        color: #d4f0fc;
        border: 1px solid #3a9bc8;
        transition: all 0.3s ease;
    }
    
    .nav-btn:hover {
        background-color: #3a9bc8;
        color: #0c1445;
        transform: translateY(-2px);
    }
    
    .nav-btn.active {
        background-color: #3a9bc8;
        color: #0c1445;
        font-weight: bold;
    }
    
    /* Success, info, warning, error boxes */
    .stAlert {
        background-color: #1d2d50;
        border-color: #3a9bc8;
    }
    
    /* Metric cards */
    .stMetric {
        background-color: #1d2d50;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #3a9bc8;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1d2d50;
        color: #d4f0fc;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3a9bc8;
        color: #0c1445;
    }
</style>
""", unsafe_allow_html=True)

# --- Project Overview ---
PROJECT_OVERVIEW = """
<div style="background-color: #1d2d50; padding: 20px; border-radius: 10px; border-left: 5px solid #3a9bc8;">
    <h3 style="color: #5fcde4;">Traffic Accident Severity Prediction System</h3>
    <p>Traffic accidents are a major problem worldwide, causing fatalities, property damage, and productivity loss. This system uses machine learning to analyze traffic data and predict accident severity based on factors like weather, road conditions, and vehicle types.</p>
    <p style="color: #a6f7d1;"><b>Key Benefits:</b> Improved emergency response planning, better road safety measures, and data-driven accident prevention strategies.</p>
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

# --- Normalize Text Values ---
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

# --- Load Dataset ---
@st.cache_data(persist="disk")
def load_default_data():
    url = st.session_state.default_dataset
    df = pd.read_csv(url)
    return preprocess_data(df)

def preprocess_data(df):
    # Basic preprocessing
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    
    # Try to identify location data
    location_col = None
    for col in df.columns:
        if 'location' in col.lower() or 'lat' in col.lower() or 'long' in col.lower():
            location_col = col
            break
    
    if location_col:
        df['Location_Original'] = df[location_col]  # Preserve original for mapping
        # Extract coordinates from Location column if needed
        if 'Location' in df.columns:
            try:
                coords = df['Location'].str.extract(r'\(([^,]+),\s*([^)]+)\)')
                df['latitude'] = pd.to_numeric(coords[0], errors='coerce')
                df['longitude'] = pd.to_numeric(coords[1], errors='coerce')
                df.dropna(subset=['latitude', 'longitude'], inplace=True)
            except:
                pass
    
    # Try to identify target column
    target_col = st.session_state.target_col
    if target_col not in df.columns:
        # Try to find similar column
        possible_targets = [col for col in df.columns if 'severity' in col.lower() or 'injury' in col.lower()]
        if possible_targets:
            target_col = possible_targets[0]
            st.session_state.target_col = target_col
    
    # Normalize categories with empty custom mappings (use defaults)
    df = normalize_categories(df, custom_mappings={})
    
    # Encode categorical columns
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
    # Try to remove location columns if they exist
    loc_cols = [col for col in X.columns if 'location' in col.lower() or col in ['latitude', 'longitude']]
    if loc_cols:
        X = X.drop(loc_cols, axis=1)
    
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X, y, X_train, X_test, y_train, y_test

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

# --- Admin Page Functions ---
def handle_dataset_upload(uploaded_file):
    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Read the CSV file
        new_df = pd.read_csv(tmp_path)
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        # Preprocess the new dataset
        new_df, new_label_encoders, new_target_col = preprocess_data(new_df)
        new_X, new_y, new_X_train, new_X_test, new_y_train, new_y_test = prepare_model_data(new_df, new_target_col)
        
        # Train models on new data
        new_models, new_scores_df = train_models(new_X_train, new_y_train, new_X_test, new_y_test)
        
        # Update session state
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

# --- Navigation Buttons ---
def navigation():
    st.sidebar.title("🚗 Accident Severity Predictor")
    st.sidebar.markdown("---")
    
    # Navigation buttons with icons
    if st.sidebar.button("🏠 Home", key="nav_home", use_container_width=True):
        st.session_state.current_page = "Home"
    if st.sidebar.button("📊 Data Analysis", key="nav_analysis", use_container_width=True):
        st.session_state.current_page = "Data Analysis"
    if st.sidebar.button("🔮 Prediction", key="nav_prediction", use_container_width=True):
        st.session_state.current_page = "Prediction"
    if st.sidebar.button("📝 Reports", key="nav_reports", use_container_width=True):
        st.session_state.current_page = "Reports"
    if st.sidebar.button("❓ Help", key="nav_help", use_container_width=True):
        st.session_state.current_page = "Help"
    
    # Admin mode toggle
    st.sidebar.markdown("---")
    admin_mode = st.sidebar.checkbox("🔒 Admin Mode")
    
    if admin_mode:
        if st.sidebar.button("⚙️ Admin Dashboard", key="nav_admin", use_container_width=True):
            st.session_state.current_page = "Admin"
    
    # Add some styling to the sidebar
    st.sidebar.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #0c1445;
        }
        [data-testid="stSidebar"] {
            background-color: #0c1445;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Home Page ---
def home_page():
    st.title("Traffic Accident Severity Prediction")
    st.markdown(PROJECT_OVERVIEW, unsafe_allow_html=True)
    
    with st.expander("📁 Dataset Overview", expanded=True):
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.current_df.copy().head().style.set_properties(**{
            'background-color': '#1d2d50',
            'color': '#d4f0fc',
            'border-color': '#3a9bc8'
        }))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", len(st.session_state.current_df))
        with col2:
            st.metric("Number of Features", len(st.session_state.current_df.columns))
    
    st.markdown("---")
    st.subheader("🚀 Quick Start Guide")
    st.markdown("""
    1. **Data Analysis**: Explore visualizations and insights
    2. **Prediction**: Make custom severity predictions
    3. **Reports**: View detailed dataset information
    """)

# --- Data Analysis Page ---
def data_analysis_page():
    st.title("📊 Data Analysis & Insights")
    st.markdown("*Explore key patterns and model performance.*")
    st.markdown("---")
    
    # Get current data from session state
    df = st.session_state.current_df
    scores_df = st.session_state.scores_df
    
    with st.expander("🎯 Target Variable Distribution", expanded=True):
        if st.session_state.target_col in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x=st.session_state.target_col, data=df, ax=ax, palette=PALETTE)
            
            # Add severity level labels
            severity_labels = {
                0: "No Injury",
                1: "Minor Injury",
                2: "Moderate Injury",
                3: "Serious Injury",
                4: "Fatal Injury"
            }
            
            # Get current labels and replace with severity labels if they match
            current_labels = [int(tick.get_text()) for tick in ax.get_xticklabels()]
            new_labels = [severity_labels.get(label, label) for label in current_labels]
            ax.set_xticklabels(new_labels, rotation=45, ha='right')
            
            ax.set_title(f'Count of {st.session_state.target_col} Levels', color='#d4f0fc')
            ax.set_xlabel('Severity Level', color='#d4f0fc')
            ax.set_ylabel('Count', color='#d4f0fc')
            st.pyplot(fig)
        else:
            st.warning(f"Target column '{st.session_state.target_col}' not found in dataset.")
    
    with st.expander("📍 Accident Hotspot Locations", expanded=True):
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Create Folium map with dark tiles
            m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], 
                          zoom_start=11, 
                          tiles='CartoDB dark_matter',
                          attr='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>')
            
            # Add heatmap
            from folium.plugins import HeatMap
            heat_data = [[row['latitude'], row['longitude']] for idx, row in df.iterrows()]
            HeatMap(heat_data, radius=10).add_to(m)
            
            folium_static(m, width=1000, height=600)
        else:
            st.warning("No location data found in dataset.")
    
    with st.expander("📈 Feature Correlation", expanded=True):
        try:
            corr = df.select_dtypes(['number']).corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax, center=0,
                       cbar_kws={'label': 'Correlation Coefficient'})
            ax.set_title("Feature Correlation Heatmap", color='#d4f0fc', pad=20)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate correlation heatmap: {str(e)}")
    
    if not st.session_state.scores_df.empty:
        with st.expander("🤖 Model Performance", expanded=True):
            st.subheader("Model Performance Metrics")
            # Format the scores to show 2 decimal places
            formatted_scores = st.session_state.scores_df.copy()
            for col in formatted_scores.columns[1:]:
                formatted_scores[col] = formatted_scores[col].apply(lambda x: f"{x:.2f}")
            
            # Display as a styled table
            st.table(formatted_scores.style.set_properties(**{
                'background-color': '#1d2d50',
                'color': '#d4f0fc',
                'border-color': '#3a9bc8'
            }))
            
            st.subheader("Performance Comparison")
            performance_df = st.session_state.scores_df.set_index('Model')
            fig, ax = plt.subplots(figsize=(10, 6))
            performance_df.plot(kind='bar', ax=ax, color=PALETTE)
            ax.set_title('Model Performance Comparison', color='#d4f0fc', pad=20)
            ax.set_ylabel('Score (%)', color='#d4f0fc')
            ax.set_xlabel('Model', color='#d4f0fc')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(facecolor='#0c1445', edgecolor='#0c1445')
            st.pyplot(fig)
            
            st.subheader("Feature Importance Analysis")
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
                    
                    # Ensure we don't try to access more features than available
                    n_features = min(10, len(st.session_state.X.columns))
                    top_features = st.session_state.X.columns[sorted_idx][:n_features]
                    top_vals = importances_vals[sorted_idx][:n_features]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=top_vals, y=top_features, ax=ax, palette=PALETTE)
                    ax.set_title(f'{model_name} - Top {n_features} Features', color='#d4f0fc', pad=20)
                    ax.set_xlabel('Importance Score', color='#d4f0fc')
                    ax.set_ylabel('Feature', color='#d4f0fc')
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not display feature importances: {str(e)}")

# --- Prediction Page ---
def prediction_page():
    st.title("🔮 Accident Severity Prediction")
    st.markdown("*Make custom predictions by selecting input values below.*")
    st.markdown("---")
    
    if not st.session_state.models:
        st.warning("No models available for prediction. Please check the Data Analysis page.")
    else:
        selected_model = st.selectbox("Select Prediction Model", list(st.session_state.models.keys()))
        model = st.session_state.models[selected_model]
        
        with st.expander("⚙️ Input Parameters", expanded=True):
            col1, col2 = st.columns(2)
            
            input_data = {}
            for i, col in enumerate(st.session_state.X.columns):
                # Alternate between columns
                current_col = col1 if i % 2 == 0 else col2
                
                if col in st.session_state.label_encoders:
                    options = sorted(st.session_state.label_encoders[col].classes_)
                    choice = current_col.selectbox(f"{col}", options)
                    input_data[col] = st.session_state.label_encoders[col].transform([choice])[0]
                else:
                    # Get min/max from the original dataframe (before scaling)
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
        
        if st.button("🚦 Predict Severity", key="predict_button", use_container_width=True):
            input_df = pd.DataFrame([input_data])
            try:
                prediction = model.predict(input_df)[0]
                probs = model.predict_proba(input_df)[0]
                confidence = np.max(probs) * 100

                if st.session_state.target_col in st.session_state.label_encoders:
                    severity_label = st.session_state.label_encoders[st.session_state.target_col].inverse_transform([prediction])[0]
                else:
                    severity_label = prediction

                # Display prediction results
                st.success("Prediction Complete!")
                
                # Create columns for better layout
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric(label="Predicted Severity", value=severity_label)
                
                with res_col2:
                    st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
                
                # Show probability distribution
                if st.session_state.target_col in st.session_state.label_encoders:
                    st.subheader("Probability Distribution")
                    prob_df = pd.DataFrame({
                        'Severity Level': st.session_state.label_encoders[st.session_state.target_col].classes_,
                        'Probability': probs * 100
                    })
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(x='Severity Level', y='Probability', data=prob_df, 
                                ax=ax, palette=PALETTE)
                    ax.set_title('Severity Probability Distribution', color='#d4f0fc')
                    ax.set_xlabel('Severity Level', color='#d4f0fc')
                    ax.set_ylabel('Probability (%)', color='#d4f0fc')
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# --- Reports Page ---
def reports_page():
    st.title("📝 Dataset Reports")
    st.markdown("---")
    
    with st.expander("📊 Summary Statistics", expanded=True):
        st.dataframe(st.session_state.current_df.describe().style.set_properties(**{
            'background-color': '#1d2d50',
            'color': '#d4f0fc',
            'border-color': '#3a9bc8'
        }))
    
    with st.expander("📋 Column Information", expanded=True):
        col_info = pd.DataFrame({
            'Column': st.session_state.current_df.columns,
            'Data Type': st.session_state.current_df.dtypes,
            'Unique Values': [st.session_state.current_df[col].nunique() for col in st.session_state.current_df.columns]
        })
        st.dataframe(col_info.style.set_properties(**{
            'background-color': '#1d2d50',
            'color': '#d4f0fc',
            'border-color': '#3a9bc8'
        }))
    
    with st.expander("⚠️ Data Quality Report", expanded=True):
        missing_data = st.session_state.current_df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            st.warning("Columns with missing values detected:")
            st.dataframe(missing_data.reset_index().rename(columns={'index': 'Column', 0: 'Missing Values'}))
        else:
            st.success("No missing values found in the dataset.")

# --- Help Page ---
def help_page():
    st.title("❓ User Guide")
    st.markdown("---")
    
    with st.expander("📌 Application Overview", expanded=True):
        st.markdown("""
        This application provides tools for analyzing traffic accident data and predicting accident severity using machine learning models.
        
        ### Key Features:
        - **Interactive data visualizations**
        - **Multiple machine learning models**
        - **Custom prediction interface**
        - **Comprehensive reporting**
        """)
    
    with st.expander("🗺️ Navigation Guide", expanded=True):
        st.markdown("""
        Use the sidebar to navigate between different sections:
        
        - **🏠 Home**: Project overview and dataset preview
        - **📊 Data Analysis**: Visualizations and insights from the data
        - **🔮 Prediction**: Make custom severity predictions
        - **📝 Reports**: View detailed dataset information
        - **❓ Help**: This user guide
        """)
    
    with st.expander("🔍 Data Analysis Section", expanded=True):
        st.markdown("""
        The Data Analysis page includes:
        - 🎯 Target variable distribution
        - 📍 Accident location hotspots
        - 📈 Feature correlation analysis
        - 🤖 Model performance metrics
        - 🔑 Feature importance analysis
        """)
    
    with st.expander("⚡ Prediction Section", expanded=True):
        st.markdown("""
        To make predictions:
        1. Select a machine learning model
        2. Set the input parameters
        3. Click "🚦 Predict Severity"
        4. View the results
        
        The system will show:
        - Predicted severity level
        - Confidence score
        - Probability distribution across all severity levels
        """)

# --- Admin Page ---
def admin_page():
    st.title("⚙️ Administration Dashboard")
    
    # Simple password protection
    password = st.text_input("Enter Admin Password:", type="password", key="admin_password")
    
    if password != "admin123":
        st.error("Incorrect password. Access denied.")
        st.stop()  # This stops execution if password is wrong
    
    st.warning("⚠️ You are in administrator mode. Changes here will affect all users.")
    st.markdown("---")
    
    with st.expander("📁 Dataset Management", expanded=True):
        uploaded_file = st.file_uploader("Upload New Dataset (CSV)", type="csv", key="dataset_uploader")
        
        if uploaded_file is not None:
            st.info("File uploaded successfully. Click the button below to update the system.")
            if st.button("🔄 Update System with New Dataset", key="update_dataset"):
                with st.spinner("Processing new dataset and retraining models..."):
                    handle_dataset_upload(uploaded_file)
    
    with st.expander("📊 System Information", expanded=True):
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.metric("Current Target Variable", st.session_state.target_col)
            st.metric("Number of Features", len(st.session_state.X.columns) if st.session_state.X is not None else 0)
        
        with info_col2:
            st.metric("Number of Models", len(st.session_state.models))
            st.metric("Dataset Rows", len(st.session_state.current_df))
    
    with st.expander("⚙️ System Maintenance", expanded=True):
        if st.button("🔄 Reset to Default Dataset", key="reset_system"):
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

# --- Main App ---
def main():
    navigation()
    
    # Page routing
    if st.session_state.current_page == "Home":
        home_page()
    elif st.session_state.current_page == "Data Analysis":
        data_analysis_page()
    elif st.session_state.current_page == "Prediction":
        prediction_page()
    elif st.session_state.current_page == "Reports":
        reports_page()
    elif st.session_state.current_page == "Help":
        help_page()
    elif st.session_state.current_page == "Admin":
        admin_page()

if __name__ == "__main__":
    main()
