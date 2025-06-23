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

# --- Config ---
st.set_page_config(page_title="Accident Severity Predictor", layout="wide")
sns.set_style("whitegrid")
PALETTE = sns.color_palette("Set2")

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

# --- Side Menu ---
st.sidebar.title("Navigation")
if 'admin_mode' not in st.session_state:
    st.session_state.admin_mode = False

if st.sidebar.checkbox("Admin Mode"):
    st.session_state.admin_mode = True
    page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Prediction", "Reports", "Help", "Admin"])
else:
    st.session_state.admin_mode = False
    page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Prediction", "Reports", "Help"])

# --- Home ---
if page == "Home":
    st.title("Traffic Accident Severity Prediction")
    st.write(PROJECT_OVERVIEW)

    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.current_df.copy().head())

# --- Data Analysis ---
elif page == "Data Analysis":
    st.title("Data Analysis & Insights")
    st.markdown("*Explore key patterns and model performance.*")
    st.divider()

    # Get current data from session state
    df = st.session_state.current_df
    scores_df = st.session_state.scores_df
    
    st.subheader("➥ Target Variable Distribution")
    if st.session_state.target_col in df.columns:
        fig, ax = plt.subplots()
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
        ax.set_xticklabels(new_labels)
        
        ax.set_title(f'Count of {st.session_state.target_col} Levels')
        ax.set_xlabel('Severity Level')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    else:
        st.warning(f"Target column '{st.session_state.target_col}' not found in dataset.")
    st.divider()

    st.subheader("➥ Hotspot Location")
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Create Folium map with dark tiles
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], 
                      zoom_start=11, 
                      tiles='CartoDB dark_matter')
        
        # Add points to the map
        for idx, row in df.sample(min(1000, len(df))).iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7
            ).add_to(m)
        
        folium_static(m, width=1000, height=600)
    else:
        # Try to find latitude/longitude columns
        lat_col = next((col for col in df.columns if 'lat' in col.lower()), None)
        long_col = next((col for col in df.columns if 'long' in col.lower()), None)
        
        if lat_col and long_col:
            try:
                # Create Folium map with dark tiles
                m = folium.Map(location=[df[lat_col].mean(), df[long_col].mean()], 
                              zoom_start=11, 
                              tiles='CartoDB dark_matter')
                
                # Add points to the map
                for idx, row in df.sample(min(1000, len(df))).iterrows():
                    folium.CircleMarker(
                        location=[row[lat_col], row[long_col]],
                        radius=3,
                        color='red',
                        fill=True,
                        fill_color='red',
                        fill_opacity=0.7
                    ).add_to(m)
                
                folium_static(m, width=1000, height=600)
            except:
                st.warning("Could not create map with available coordinates.")
        else:
            st.warning("No location data found in dataset.")
    st.divider()

    st.subheader("➥ Correlation Heatmap")
    try:
        corr = df.select_dtypes(['number']).corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap='YlGnBu', annot=False, ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    except:
        st.warning("Could not generate correlation heatmap.")
    st.divider()

    if not st.session_state.scores_df.empty:
        st.subheader("➥ Model Performance")
        st.table(st.session_state.scores_df.round(2))
        st.divider()

        st.subheader("➥ Model Comparison Bar Chart")
        performance_df = st.session_state.scores_df.set_index('Model')
        fig, ax = plt.subplots()
        performance_df.plot(kind='bar', ax=ax, color=PALETTE.as_hex())
        ax.set_title('Model Comparison')
        ax.set_ylabel('Score (%)')
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
        st.divider()

        st.subheader("➥ Model-Specific Feature Importances")
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

                fig, ax = plt.subplots()
                sns.barplot(x=top_vals, y=top_features, ax=ax, palette=PALETTE)
                ax.set_title(f'{model_name} Top {n_features} Features')
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not display feature importances: {str(e)}")

# --- Prediction ---
elif page == "Prediction":
    st.title("Custom Prediction")
    
    if not st.session_state.models:
        st.warning("No models available for prediction. Please check the Data Analysis page.")
    else:
        selected_model = st.selectbox("Choose Model for Prediction", list(st.session_state.models.keys()))
        model = st.session_state.models[selected_model]
        
        input_data = {}
        for col in st.session_state.X.columns:
            if col in st.session_state.label_encoders:
                options = sorted(st.session_state.label_encoders[col].classes_)
                choice = st.selectbox(f"{col}", options)
                input_data[col] = st.session_state.label_encoders[col].transform([choice])[0]
            else:
                # Get min/max from the original dataframe (before scaling)
                col_min = st.session_state.current_df[col].min()
                col_max = st.session_state.current_df[col].max()
                col_mean = st.session_state.current_df[col].mean()
                input_data[col] = st.number_input(f"{col}", float(col_min), float(col_max), float(col_mean))
        
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            try:
                prediction = model.predict(input_df)[0]
                probs = model.predict_proba(input_df)[0]
                confidence = np.max(probs) * 100

                if st.session_state.target_col in st.session_state.label_encoders:
                    severity_label = st.session_state.label_encoders[st.session_state.target_col].inverse_transform([prediction])[0]
                else:
                    severity_label = prediction

                st.success(f"**Predicted {st.session_state.target_col}:** {severity_label}")
                st.info(f"**Confidence:** {confidence:.2f}%")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# --- Reports ---
elif page == "Reports":
    st.title("Dataset Reports")
    st.write("### Dataset Summary")
    st.dataframe(st.session_state.current_df.describe())
    
    st.write("### Column Information")
    col_info = pd.DataFrame({
        'Column': st.session_state.current_df.columns,
        'Data Type': st.session_state.current_df.dtypes,
        'Unique Values': [st.session_state.current_df[col].nunique() for col in st.session_state.current_df.columns]
    })
    st.dataframe(col_info)

# --- Help Page ---
elif page == "Help":
    st.title("User Manual")
    
    st.markdown("""
    ### How to Use This Application
    
    **Navigation:**
    - Use the sidebar to navigate between different sections of the application.
    
    **Pages:**
    
    **1. Home**
    - Overview of the project
    - Preview of the dataset
    
    **2. Data Analysis**
    - Visualizations of the target variable distribution
    - Accident hotspot map
    - Correlation heatmap
    - Model performance metrics
    - Feature importance analysis
    
    **3. Prediction**
    - Make custom predictions by selecting input values
    - Choose between different machine learning models
    - View prediction confidence levels
    
    **4. Reports**
    - View dataset statistics
    - See column information and data types
    
    **5. Admin (Admin Mode Only)**
    - Upload new datasets
    - Reset to default dataset
    - View system information
    
    **Admin Access:**
    - Enable Admin Mode in the sidebar
    - Password: admin1
    """)

# --- Admin Page ---
elif page == "Admin":
    st.title("Admin Dashboard")
    
    # Simple password protection
    password = st.text_input("Enter Admin Password:", type="password")
    
    if password != "admin1":
        st.error("Incorrect password. Access denied.")
        st.stop()  # This stops execution if password is wrong
    
    st.warning("You are in admin mode. Changes here will affect all users.")
    
    st.subheader("Upload New Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        st.info("File uploaded successfully. Click the button below to update the system.")
        if st.button("Update System with New Dataset"):
            with st.spinner("Processing new dataset and retraining models..."):
                handle_dataset_upload(uploaded_file)
    
    st.subheader("Current System Information")
    st.write(f"Current target variable: {st.session_state.target_col}")
    st.write(f"Number of features: {len(st.session_state.X.columns) if st.session_state.X is not None else 0}")
    st.write(f"Number of models: {len(st.session_state.models)}")
    
    st.subheader("Reset to Default Dataset")
    if st.button("Reset System"):
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
            
            st.success("System reset to default dataset!")
