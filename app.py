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

# --- Custom Dark Theme with Coolwarm Palette ---
def set_dark_theme():
    # Set seaborn style
    sns.set_style("darkgrid")
    
    # Coolwarm color palette
    PALETTE = sns.color_palette("coolwarm", 10)
    
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

PALETTE = set_dark_theme()

# --- Streamlit Config ---
st.set_page_config(page_title="Accident Severity Predictor", layout="wide")

# Custom CSS for modern dark theme
st.markdown(f"""
<style>
    /* Main page background */
    .stApp {{
        background-color: #0E1117;
        color: white;
    }}
    
    /* Sidebar styling */
    .css-1d391kg {{
        background-color: #121721;
        border-right: 1px solid #2A3459;
    }}
    
    /* Radio button styling */
    .stRadio > div {{
        background-color: #1E2130;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #2A3459;
    }}
    
    .stRadio > label {{
        color: white !important;
    }}
    
    /* Selected radio button */
    .stRadio > div > div > div > div {{
        background-color: #3A4A6B !important;
        color: white !important;
    }}
    
    /* Widgets */
    .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak, .st-al, .st-am, .st-an, .st-ao, .st-ap, .st-aq, .st-ar, .st-as {{
        background-color: #1E2130;
        color: white;
        border-color: #2A3459;
    }}
    
    /* Text input */
    .stTextInput input {{
        color: white !important;
    }}
    
    /* Select boxes */
    .stSelectbox select {{
        color: white !important;
    }}
    
    /* Number input */
    .stNumberInput input {{
        color: white !important;
    }}
    
    /* Dataframes */
    .stDataFrame {{
        background-color: #1E2130;
    }}
    
    /* Tables */
    table {{
        color: white !important;
    }}
    
    /* Markdown text color */
    .stMarkdown {{
        color: white;
    }}
    
    /* Divider color */
    hr {{
        border-color: #3A4A6B;
        margin: 1.5rem 0;
    }}
    
    /* Button styling */
    .stButton>button {{
        background-color: #3A4A6B;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        background-color: #4D5F8B;
        color: white;
    }}
    
    /* Cards */
    .card {{
        background-color: #1E2130;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #2A3459;
    }}
    
    /* Titles */
    h1, h2, h3, h4, h5, h6 {{
        color: #6C9BCF;
    }}
</style>
""", unsafe_allow_html=True)

# --- Project Overview ---
PROJECT_OVERVIEW = """
<div class="card">
    <h3>Traffic Accident Severity Prediction System</h3>
    <p>This advanced analytics platform leverages machine learning to predict accident severity based on environmental, 
    vehicular, and human factors. The system helps authorities prioritize emergency responses and develop targeted 
    safety interventions.</p>
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
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
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

# --- Modern Navigation Panel ---
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

# Navigation options with icons (using text)
nav_options = {
    "Dashboard": "📊",
    "Data Explorer": "🔍",
    "Predictor": "🔮",
    "Reports": "📋",
    "Help Center": "❓"
}

if st.sidebar.checkbox("Admin Mode", key="admin_mode"):
    nav_options["Admin Console"] = "⚙️"

# Create radio buttons with custom styling
selected_page = st.sidebar.radio(
    "Select Page",
    list(nav_options.keys()),
    format_func=lambda x: f"{nav_options[x]} {x}",
    key="nav_radio"
)

# Map navigation options to actual pages
page_mapping = {
    "Dashboard": "Home",
    "Data Explorer": "Data Analysis",
    "Predictor": "Prediction",
    "Reports": "Reports",
    "Help Center": "Help",
    "Admin Console": "Admin"
}

page = page_mapping[selected_page]

# --- Home ---
if page == "Home":
    st.title("Accident Severity Analytics Dashboard")
    st.markdown(PROJECT_OVERVIEW)

    st.subheader("Dataset Overview")
    with st.expander("View Dataset Sample"):
        st.dataframe(st.session_state.current_df.copy().head(10).style.set_properties(**{
            'background-color': '#1E2130',
            'color': 'white',
            'border-color': '#2A3459'
        }))

    st.subheader("Quick Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(st.session_state.current_df))
    
    with col2:
        st.metric("Features Available", len(st.session_state.current_df.columns))
    
    with col3:
        if st.session_state.target_col in st.session_state.current_df.columns:
            unique_classes = len(st.session_state.current_df[st.session_state.target_col].unique())
            st.metric("Severity Classes", unique_classes)
        else:
            st.metric("Target Variable", "Not Found")

# --- Data Analysis ---
elif page == "Data Analysis":
    st.title("Data Explorer")
    st.markdown("*Interactive visualizations and model performance metrics*")
    
    # Get current data from session state
    df = st.session_state.current_df
    scores_df = st.session_state.scores_df
    
    with st.container():
        st.subheader("Severity Distribution")
        if st.session_state.target_col in df.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
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
            
            ax.set_title(f'Severity Level Distribution', color='white')
            ax.set_xlabel('Severity Level', color='white')
            ax.set_ylabel('Count', color='white')
            st.pyplot(fig)
        else:
            st.warning(f"Target column '{st.session_state.target_col}' not found.")
    
    st.divider()
    
    with st.container():
        st.subheader("Geospatial Hotspots")
        if 'latitude' in df.columns and 'longitude' in df.columns:
            m = folium.Map(
                location=[df['latitude'].mean(), df['longitude'].mean()], 
                zoom_start=11, 
                tiles='CartoDB dark_matter',
                control_scale=True
            )
            
            # Add heatmap
            from folium.plugins import HeatMap
            heat_data = [[row['latitude'], row['longitude']] for _, row in df.iterrows()]
            HeatMap(heat_data, radius=10).add_to(m)
            
            folium_static(m, width=1000, height=500)
        else:
            st.warning("Geospatial data not available in this dataset.")
    
    st.divider()
    
    with st.container():
        st.subheader("Feature Relationships")
        tab1, tab2 = st.tabs(["Correlation Matrix", "Feature Importance"])
        
        with tab1:
            try:
                corr = df.select_dtypes(['number']).corr()
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax, center=0,
                           cbar_kws={'label': 'Correlation Coefficient'})
                ax.set_title("Feature Correlation Matrix", color='white', pad=20)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate correlation matrix: {str(e)}")
        
        with tab2:
            if not st.session_state.scores_df.empty:
                model_name = st.selectbox(
                    "Select Model for Feature Importance", 
                    list(st.session_state.models.keys()),
                    index=1
                )
                
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

                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.barplot(x=top_vals, y=top_features, ax=ax, palette='coolwarm')
                        ax.set_title(f'Top {n_features} Features - {model_name}', color='white')
                        ax.set_xlabel('Importance Score', color='white')
                        ax.set_ylabel('Feature', color='white')
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Feature importance not available: {str(e)}")
    
    st.divider()
    
    with st.container():
        st.subheader("Model Performance")
        if not st.session_state.scores_df.empty:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(
                    st.session_state.scores_df.style.format({
                        'Accuracy (%)': '{:.2f}',
                        'Precision (%)': '{:.2f}',
                        'Recall (%)': '{:.2f}',
                        'F1-Score (%)': '{:.2f}'
                    }).set_properties(**{
                        'background-color': '#1E2130',
                        'color': 'white',
                        'border-color': '#2A3459'
                    })
                )
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 5))
                melted_df = st.session_state.scores_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
                sns.barplot(x='Model', y='Score', hue='Metric', data=melted_df, ax=ax, palette='coolwarm')
                ax.set_title('Model Performance Comparison', color='white')
                ax.set_xlabel('Model', color='white')
                ax.set_ylabel('Score (%)', color='white')
                ax.legend(facecolor='#0E1117', edgecolor='#0E1117')
                st.pyplot(fig)

# --- Prediction ---
elif page == "Prediction":
    st.title("Accident Severity Predictor")
    st.markdown("*Predict potential accident severity based on input parameters*")
    
    if not st.session_state.models:
        st.warning("No trained models available. Please check the Data Explorer page.")
    else:
        with st.container():
            st.subheader("Model Selection")
            selected_model = st.selectbox(
                "Choose Prediction Model", 
                list(st.session_state.models.keys()),
                help="Select the machine learning model to use for predictions"
            )
            model = st.session_state.models[selected_model]
        
        with st.container():
            st.subheader("Input Parameters")
            col1, col2 = st.columns(2)
            
            input_data = {}
            for i, col in enumerate(st.session_state.X.columns):
                current_col = col1 if i % 2 == 0 else col2
                
                if col in st.session_state.label_encoders:
                    options = sorted(st.session_state.label_encoders[col].classes_)
                    choice = current_col.selectbox(
                        f"{col}", 
                        options,
                        help=f"Select value for {col}"
                    )
                    input_data[col] = st.session_state.label_encoders[col].transform([choice])[0]
                else:
                    col_min = st.session_state.current_df[col].min()
                    col_max = st.session_state.current_df[col].max()
                    col_mean = st.session_state.current_df[col].mean()
                    input_data[col] = current_col.slider(
                        f"{col}", 
                        float(col_min), 
                        float(col_max), 
                        float(col_mean),
                        help=f"Adjust value for {col} (range: {col_min:.2f} to {col_max:.2f})"
                    )
        
        if st.button("Predict Severity", type="primary"):
            with st.spinner("Processing prediction..."):
                input_df = pd.DataFrame([input_data])
                try:
                    prediction = model.predict(input_df)[0]
                    probs = model.predict_proba(input_df)[0]
                    confidence = np.max(probs) * 100

                    if st.session_state.target_col in st.session_state.label_encoders:
                        severity_label = st.session_state.label_encoders[st.session_state.target_col].inverse_transform([prediction])[0]
                    else:
                        severity_label = prediction

                    st.success("Prediction Complete!")
                    
                    # Display results in cards
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.container():
                            st.markdown("""
                            <div class="card">
                                <h3>Prediction Result</h3>
                                <h1 style="color: #6C9BCF; text-align: center;">{}</h1>
                            </div>
                            """.format(severity_label), unsafe_allow_html=True)
                    
                    with col2:
                        with st.container():
                            st.markdown("""
                            <div class="card">
                                <h3>Confidence Level</h3>
                                <h1 style="color: #6C9BCF; text-align: center;">{:.2f}%</h1>
                            </div>
                            """.format(confidence), unsafe_allow_html=True)
                    
                    # Show probability distribution
                    if st.session_state.target_col in st.session_state.label_encoders:
                        st.subheader("Probability Breakdown")
                        prob_df = pd.DataFrame({
                            'Severity Level': st.session_state.label_encoders[st.session_state.target_col].classes_,
                            'Probability (%)': probs * 100
                        })
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        sns.barplot(x='Severity Level', y='Probability (%)', data=prob_df, 
                                    ax=ax, palette='coolwarm')
                        ax.set_title('Prediction Probability Distribution', color='white')
                        ax.set_xlabel('Severity Level', color='white')
                        ax.set_ylabel('Probability (%)', color='white')
                        st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

# --- Reports ---
elif page == "Reports":
    st.title("Data Reports")
    
    with st.container():
        st.subheader("Dataset Summary")
        st.dataframe(
            st.session_state.current_df.describe().style.format("{:.2f}").set_properties(**{
                'background-color': '#1E2130',
                'color': 'white',
                'border-color': '#2A3459'
            })
        )
    
    st.divider()
    
    with st.container():
        st.subheader("Data Dictionary")
        col_info = pd.DataFrame({
            'Column': st.session_state.current_df.columns,
            'Data Type': st.session_state.current_df.dtypes,
            'Unique Values': [st.session_state.current_df[col].nunique() for col in st.session_state.current_df.columns],
            'Missing Values': st.session_state.current_df.isnull().sum()
        })
        st.dataframe(
            col_info.style.set_properties(**{
                'background-color': '#1E2130',
                'color': 'white',
                'border-color': '#2A3459'
            })
        )
    
    st.divider()
    
    with st.container():
        st.subheader("Data Quality Report")
        missing_data = st.session_state.current_df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="card">
                    <h3>Missing Values</h3>
                    <p>The following columns contain missing data:</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.dataframe(
                    missing_data.reset_index().rename(columns={'index': 'Column', 0: 'Missing Values'}).style.set_properties(**{
                        'background-color': '#1E2130',
                        'color': 'white',
                        'border-color': '#2A3459'
                    })
                )
        else:
            st.success("No missing values detected in the dataset.")

# --- Help Page ---
elif page == "Help":
    st.title("Help Center")
    
    with st.container():
        st.subheader("Application Guide")
        st.markdown("""
        <div class="card">
            <h3>Getting Started</h3>
            <p>This application provides tools for analyzing traffic accident data and predicting 
            accident severity using machine learning models.</p>
            
            <h3>Navigation</h3>
            <ul>
                <li><b>Dashboard</b>: Overview of the dataset and key metrics</li>
                <li><b>Data Explorer</b>: Interactive visualizations and model performance</li>
                <li><b>Predictor</b>: Make custom severity predictions</li>
                <li><b>Reports</b>: Detailed dataset documentation</li>
                <li><b>Help Center</b>: This documentation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    with st.container():
        st.subheader("Frequently Asked Questions")
        
        with st.expander("How do I make a prediction?"):
            st.markdown("""
            1. Navigate to the **Predictor** page
            2. Select a machine learning model
            3. Adjust all input parameters
            4. Click the "Predict Severity" button
            5. View your results
            """)
        
        with st.expander("What do the different severity levels mean?"):
            st.markdown("""
            - **No Injury**: Property damage only
            - **Minor Injury**: No medical attention required
            - **Moderate Injury**: Medical attention required but not life-threatening
            - **Serious Injury**: Life-threatening injuries requiring hospitalization
            - **Fatal Injury**: Resulting in death
            """)
        
        with st.expander("Which model should I use for predictions?"):
            st.markdown("""
            - **Random Forest** generally provides good balance of accuracy and interpretability
            - **XGBoost** often has the highest accuracy but can be slower
            - **Logistic Regression** is fastest but may be less accurate for complex patterns
            - Compare model performance in the **Data Explorer** page
            """)

# --- Admin Page ---
elif page == "Admin":
    st.title("Administration Console")
    
    # Password protection
    password = st.text_input("Enter Admin Password:", type="password", key="admin_password")
    
    if password != "admin1":
        st.error("Incorrect password. Access denied.")
        st.stop()
    
    with st.container():
        st.warning("""
        <div style="background-color: #3A4A6B; padding: 1rem; border-radius: 8px;">
            <h3 style="color: white;">Administrator Mode</h3>
            <p style="color: white;">You are making changes to the system configuration.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    with st.container():
        st.subheader("Dataset Management")
        
        with st.expander("Upload New Dataset"):
            uploaded_file = st.file_uploader(
                "Select CSV Dataset", 
                type="csv",
                help="Upload a new dataset to replace the current one"
            )
            
            if uploaded_file is not None:
                st.info("New dataset ready for import. Click below to update system.")
                if st.button("Update System with New Dataset", type="primary"):
                    with st.spinner("Processing new dataset..."):
                        handle_dataset_upload(uploaded_file)
    
    st.divider()
    
    with st.container():
        st.subheader("System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>Current Configuration</h3>
                <p><b>Target Variable:</b> {}</p>
                <p><b>Dataset Rows:</b> {:,}</p>
                <p><b>Features:</b> {}</p>
            </div>
            """.format(
                st.session_state.target_col,
                len(st.session_state.current_df),
                len(st.session_state.current_df.columns)
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h3>Model Status</h3>
                <p><b>Trained Models:</b> {}</p>
                <p><b>Best Accuracy:</b> {:.2f}%</p>
                <p><b>Last Update:</b> {}</p>
            </div>
            """.format(
                len(st.session_state.models),
                st.session_state.scores_df['Accuracy (%)'].max(),
                pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
            ), unsafe_allow_html=True)
    
    st.divider()
    
    with st.container():
        st.subheader("System Maintenance")
        
        if st.button("Reset to Default Dataset", type="secondary"):
            with st.spinner("Restoring default configuration..."):
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
                
                st.success("System reset complete! Using default dataset.")
