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

# --- Custom Dark Theme Configuration ---
def set_dark_theme():
    # Set seaborn style
    sns.set_style("darkgrid")

    # Using coolwarm color palette
    PALETTE = sns.color_palette("coolwarm")

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

# Custom CSS for dark theme with enhanced styling
st.markdown("""
<style>
    /* Main page background */
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #0F131D !important;
        border-right: 1px solid #2A3459;
    }
    
    /* Navigation buttons */
    .stButton>button {
        background-color: #1E2130;
        color: white;
        border-color: #2A3459;
        width: 100%;
        margin: 5px 0;
    }
    
    .stButton>button:hover {
        background-color: #2A3459;
        color: white;
    }
    
    /* Active button style */
    .stButton>button[kind="primary"] {
        background-color: #3A4D8F !important;
        font-weight: 500;
    }
    
    /* Widgets */
    .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak, .st-al, .st-am, .st-an, .st-ao, .st-ap, .st-aq, .st-ar, .st-as {
        background-color: #1E2130;
        color: white;
        border-color: #2A3459;
    }
    
    /* Text input */
    .stTextInput input {
        color: white !important;
    }
    
    /* Select boxes */
    .stSelectbox select {
        color: white !important;
    }
    
    /* Number input */
    .stNumberInput input {
        color: white !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #1E2130;
    }
    
    /* Tables */
    table {
        color: white !important;
    }
    
    /* Markdown text color */
    .stMarkdown {
        color: white;
    }
    
    /* Divider color */
    hr {
        border-color: #2A3459;
    }
    
    /* Card styling */
    .card {
        background-color: #1E2130;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #2A3459;
    }
    
    .card-title {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
        color: #4A8DF8;
    }
    
    /* Navigation button styling */
    .nav-button {
        background-color: #1E2130;
        color: white;
        border: 1px solid #2A3459;
        border-radius: 4px;
        padding: 10px 15px;
        margin: 5px 0;
        width: 100%;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .nav-button:hover {
        background-color: #2A3459;
    }
    
    .nav-button.active {
        background-color: #3A4D8F;
        font-weight: bold;
    }
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

# --- Navigation Functions ---
def navigate_to(page):
    st.session_state.current_page = page

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

# --- Page Rendering Functions ---
def render_home():
    st.title("Traffic Accident Severity Prediction")
    st.markdown(PROJECT_OVERVIEW, unsafe_allow_html=True)

    with st.expander("Dataset Preview", expanded=True):
        st.dataframe(st.session_state.current_df.copy().head().style.set_properties(**{
            'background-color': '#1E2130',
            'color': 'white',
            'border-color': '#2A3459'
        }))
    
    st.markdown("---")
    
    # Key metrics cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(st.session_state.current_df))
    with col2:
        st.metric("Features Available", len(st.session_state.current_df.columns))
    with col3:
        st.metric("Trained Models", len(st.session_state.models))

def render_data_analysis():
    st.title("Data Analysis & Insights")
    st.markdown("Explore key patterns and model performance.")

    # Get current data from session state
    df = st.session_state.current_df
    scores_df = st.session_state.scores_df

    with st.expander("Target Variable Distribution", expanded=True):
        if st.session_state.target_col in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x=st.session_state.target_col, data=df, ax=ax, palette="coolwarm")

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

            ax.set_title(f'Count of {st.session_state.target_col} Levels', color='white')
            ax.set_xlabel('Severity Level', color='white')
            ax.set_ylabel('Count', color='white')
            st.pyplot(fig)
        else:
            st.warning(f"Target column '{st.session_state.target_col}' not found in dataset.")

    with st.expander("Accident Hotspot Locations"):
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Create Folium map with dark tiles
            m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], 
                          zoom_start=11, 
                          tiles='CartoDB dark_matter',
                          attr='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>')

            # Add points to the map
            for idx, row in df.sample(min(1000, len(df))).iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    color='#ff7f0e',
                    fill=True,
                    fill_color='#ff7f0e',
                    fill_opacity=0.7
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
            # Format the scores to show 2 decimal places
            formatted_scores = st.session_state.scores_df.copy()
            for col in formatted_scores.columns[1:]:
                formatted_scores[col] = formatted_scores[col].apply(lambda x: f"{x:.2f}")

            # Display as a styled table
            st.table(formatted_scores.style.set_properties(**{
                'background-color': '#1E2130',
                'color': 'white',
                'border-color': '#2A3459'
            }))
            
            # Performance comparison chart
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
                sns.barplot(x=top_vals, y=top_features, ax=ax, palette="coolwarm")
                ax.set_title(f'{model_name} - Top {n_features} Features', color='white', pad=20)
                ax.set_xlabel('Importance Score', color='white')
                ax.set_ylabel('Feature', color='white')
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not display feature importances: {str(e)}")

def render_prediction():
    st.title("Accident Severity Prediction")
    st.markdown("Make custom predictions by selecting input values below.")

    if not st.session_state.models:
        st.warning("No models available for prediction. Please check the Data Analysis page.")
    else:
        with st.container():
            st.subheader("Model Selection")
            selected_model = st.selectbox("Select Prediction Model", list(st.session_state.models.keys()))
            model = st.session_state.models[selected_model]

        with st.container():
            st.subheader("Input Parameters")
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

                    # Display prediction results
                    with st.container():
                        st.subheader("Prediction Results")
                        res_col1, res_col2 = st.columns(2)

                        with res_col1:
                            st.markdown(f"""
                            <div class="card">
                                <div class="card-title">Predicted Severity</div>
                                <h2 style="color: #4A8DF8;">{severity_label}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with res_col2:
                            st.markdown(f"""
                            <div class="card">
                                <div class="card-title">Confidence Level</div>
                                <h2 style="color: #4A8DF8;">{confidence:.2f}%</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        # Show probability distribution
                        if st.session_state.target_col in st.session_state.label_encoders:
                            st.subheader("Probability Distribution")
                            prob_df = pd.DataFrame({
                                'Severity Level': st.session_state.label_encoders[st.session_state.target_col].classes_,
                                'Probability': probs * 100
                            })

                            fig, ax = plt.subplots(figsize=(10, 4))
                            sns.barplot(x='Severity Level', y='Probability', data=prob_df, 
                                        ax=ax, palette="coolwarm")
                            ax.set_title('Severity Probability Distribution', color='white')
                            ax.set_xlabel('Severity Level', color='white')
                            ax.set_ylabel('Probability (%)', color='white')
                            st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

def render_reports():
    st.title("Dataset Reports")

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

    with st.expander("Missing Values Report"):
        st.markdown("""
        <div class="card">
            <div class="card-title">Data Completeness</div>
            <p>Analysis of missing values in the dataset.</p>
        </div>
        """, unsafe_allow_html=True)
        missing_data = st.session_state.current_df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            st.warning("Columns with missing values:")
            st.dataframe(missing_data.reset_index().rename(columns={'index': 'Column', 0: 'Missing Values'}))
        else:
            st.success("No missing values found in the dataset.")

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

def render_admin():
    st.title("Administration Dashboard")

    # Simple password protection
    password = st.text_input("Enter Admin Password:", type="password", key="admin_password")

    if password != "admin1":
        st.error("Incorrect password. Access denied.")
        st.stop()  # This stops execution if password is wrong

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
                    handle_dataset_upload(uploaded_file)

    with st.expander("System Information"):
        st.subheader("System Information")
        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Current Target Variable</div>
                <h3>{st.session_state.target_col}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Number of Features</div>
                <h3>{len(st.session_state.X.columns) if st.session_state.X is not None else 0}</h3>
            </div>
            """, unsafe_allow_html=True)

        with info_col2:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Number of Models</div>
                <h3>{len(st.session_state.models)}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
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

# --- Sidebar Navigation ---
def create_sidebar():
    st.sidebar.title("Navigation")
    
    # Admin mode toggle
    admin_mode = st.sidebar.checkbox("Admin Mode", key="admin_mode")
    
    # Navigation buttons
    pages = ["Home", "Data Analysis", "Prediction", "Reports", "Help"]
    if admin_mode:
        pages.append("Admin")
    
    for page in pages:
        if st.sidebar.button(page, key=f"nav_{page}"):
            navigate_to(page)
    
    # Highlight the current page
    for page in pages:
        if page == st.session_state.current_page:
            st.sidebar.button(page, key=f"active_{page}", type="primary")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Info")
    st.sidebar.markdown(f"**Dataset:** {len(st.session_state.current_df)} rows")
    st.sidebar.markdown(f"**Target:** {st.session_state.target_col}")

# --- Main App ---
def main():
    create_sidebar()
    
    # Page routing
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
