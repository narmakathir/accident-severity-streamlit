# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from io import StringIO

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

# Set dark theme for visualizations
plt.style.use('dark_background')
sns.set_style("darkgrid")
PALETTE = sns.color_palette("Set2")

# --- Project Overview ---
PROJECT_OVERVIEW = """
Traffic accidents are a major problem worldwide, causing several fatalities, damage to property, and loss of productivity. Predicting accident severity based on contributors such as weather conditions, road conditions, types of vehicles, and drivers enables the authorities to take necessary actions to minimize the risk and develop better emergency responses. 
 
This project uses machine learning techniques to analyze past traffic data for accident severity prediction and present useful data to improve road safety and management.
"""

# --- Normalize Text Values ---
def normalize_categories(df):
    mappings = {
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

    for col, replacements in mappings.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col] = df[col].replace(replacements)

    return df

# --- Load Dataset ---
@st.cache_data(persist="disk")
def load_data(file_path=None):
    if file_path is None:
        url = 'https://raw.githubusercontent.com/narmakathir/accident-severity-streamlit/main/filtered_crash_data.csv'
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(file_path)
    
    # Check if required columns exist
    required_columns = ['Injury Severity']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {', '.join(missing_columns)}")
        
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    # Improved location parsing
    if 'Location' in df.columns:
        df['Location_Original'] = df['Location']
        # Extract coordinates from string if they exist
        coords = df['Location'].astype(str).str.extract(r'\(([^,]+),\s*([^)]+)\)')
        if not coords.empty:
            coords.columns = ['latitude', 'longitude']
            coords['latitude'] = pd.to_numeric(coords['latitude'], errors='coerce')
            coords['longitude'] = pd.to_numeric(coords['longitude'], errors='coerce')
            valid_coords = coords.dropna()
            if not valid_coords.empty:
                df[['latitude', 'longitude']] = coords

    df = normalize_categories(df)

    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.title()
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    target_col = 'Injury Severity'
    numeric_cols = df.select_dtypes(include='number').columns.difference([target_col])
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    X = df.drop([target_col, 'Location'], axis=1, errors='ignore')
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return df, X, y, X_train, X_test, y_train, y_test, label_encoders

# Initialize session state for data
if 'data_loaded' not in st.session_state:
    try:
        df, X, y, X_train, X_test, y_train, y_test, label_encoders = load_data()
        st.session_state.data_loaded = True
        st.session_state.df = df
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.label_encoders = label_encoders
    except Exception as e:
        st.error(f"Error loading initial data: {str(e)}")

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

if 'data_loaded' in st.session_state:
    models, scores_df = train_models(st.session_state.X_train, st.session_state.y_train, 
                                   st.session_state.X_test, st.session_state.y_test)

# --- Side Menu ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Prediction", "Reports", "Admin", "Help"])

# --- Home ---
if page == "Home":
    st.title("Traffic Accident Severity Prediction")
    st.write(PROJECT_OVERVIEW)

    if 'df' in st.session_state:
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.df.copy().head())
    else:
        st.error("No data available. Please check the dataset.")

# --- Data Analysis ---
elif page == "Data Analysis":
    st.title("Data Analysis & Insights")
    st.markdown("*Explore key patterns and model performance.*")
    st.divider()

    if 'df' not in st.session_state:
        st.error("No data available. Please check the dataset.")
        st.stop()

    st.subheader("➥ Injury Severity Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Injury Severity', data=st.session_state.df, ax=ax, palette=PALETTE)
    ax.set_title('Count of Injury Levels', color='white')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    st.pyplot(fig)
    st.divider()

    st.subheader("➥ Hotspot Location")
    if 'latitude' in st.session_state.df.columns and 'longitude' in st.session_state.df.columns:
        coords = st.session_state.df[['latitude', 'longitude']].dropna()
        if not coords.empty:
            st.map(coords)
        else:
            st.warning("No valid geographic coordinates available in the dataset.")
    else:
        st.warning("Latitude/Longitude columns not found in the dataset.")
    st.divider()

    st.subheader("➥ Correlation Heatmap")
    corr = st.session_state.df.select_dtypes(['number']).corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax)
    ax.set_title("Correlation Heatmap", color='white')
    st.pyplot(fig)
    st.divider()

    st.subheader("➥ Model Performance")
    st.table(scores_df.round(2))
    st.divider()

    st.subheader("➥ Model Comparison Bar Chart")
    performance_df = scores_df.set_index('Model')
    fig, ax = plt.subplots()
    performance_df.plot(kind='bar', ax=ax, color=PALETTE.as_hex())
    ax.set_title('Model Comparison', color='white')
    ax.set_ylabel('Score (%)', color='white')
    ax.tick_params(colors='white')
    ax.legend(title='Metrics', title_fontsize='10', fontsize='8', facecolor='black', edgecolor='white', labelcolor='white')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)
    st.divider()

    st.subheader("➥ Model-Specific Feature Importances")
    model_name = st.selectbox("Select Model", list(models.keys()), index=1)

    importances = {
        'Random Forest': models['Random Forest'].feature_importances_,
        'XGBoost': models['XGBoost'].feature_importances_,
        'Logistic Regression': np.abs(models['Logistic Regression'].coef_[0]),
        'Artificial Neural Network': np.mean(np.abs(models['Artificial Neural Network'].coefs_[0]), axis=1),
    }
    importances_vals = importances[model_name]
    importances_vals /= importances_vals.sum()

    sorted_idx = np.argsort(importances_vals)[::-1]
    top_features = st.session_state.X.columns[sorted_idx][:10]
    top_vals = importances_vals[sorted_idx][:10]

    fig, ax = plt.subplots()
    sns.barplot(x=top_vals, y=top_features, ax=ax, palette=PALETTE)
    ax.set_title(f'{model_name} Top 10 Features', color='white')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    st.pyplot(fig)

# --- Prediction ---
elif page == "Prediction":
    st.title("Custom Prediction")
    
    if 'X' not in st.session_state or 'label_encoders' not in st.session_state:
        st.error("System not properly initialized. Please check the dataset.")
        st.stop()
        
    selected_model = st.selectbox("Choose Model for Prediction", list(models.keys()))
    model = models[selected_model]

    input_data = {}
    for col in st.session_state.X.columns:
        if col in st.session_state.label_encoders:
            options = sorted(st.session_state.label_encoders[col].classes_)
            choice = st.selectbox(f"{col}", options)
            input_data[col] = st.session_state.label_encoders[col].transform([choice])[0]
        else:
            input_data[col] = st.number_input(f"{col}", float(st.session_state.df[col].min()), 
                                            float(st.session_state.df[col].max()), 
                                            float(st.session_state.df[col].mean()))

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    confidence = np.max(probs) * 100

    if 'Injury Severity' in st.session_state.label_encoders:
        severity_label = st.session_state.label_encoders['Injury Severity'].inverse_transform([prediction])[0]
    else:
        severity_label = prediction

    st.success(f"**Predicted Injury Severity:** {severity_label}")
    st.info(f"**Confidence:** {confidence:.2f}%")

# --- Reports ---
elif page == "Reports":
    st.title("Generated Reports")
    if 'df' in st.session_state:
        st.write("### Dataset Summary")
        st.dataframe(st.session_state.df.describe())
    else:
        st.error("No data available. Please check the dataset.")

# --- Admin Page ---
elif page == "Admin":
    st.title("Admin Dashboard")
    st.warning("This section is for administrators only.")
    
    password = st.text_input("Enter Admin Password:", type="password")
    if password == "admin123":  # In production, use a more secure method
        st.success("Authenticated")
        
        st.subheader("Upload New Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                new_df = pd.read_csv(stringio)
                
                # Display preview
                st.write("New Dataset Preview:")
                st.dataframe(new_df.head())
                
                if st.button("Update System with New Dataset"):
                    # Clear all caches
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    
                    # Reload data with the new file
                    with st.spinner("Updating system with new data..."):
                        # Save to temp file
                        temp_path = "temp_uploaded_data.csv"
                        new_df.to_csv(temp_path, index=False)
                        
                        try:
                            # Reload data
                            df, X, y, X_train, X_test, y_train, y_test, label_encoders = load_data(temp_path)
                            
                            # Update session state
                            st.session_state.df = df
                            st.session_state.X = X
                            st.session_state.y = y
                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                            st.session_state.y_train = y_train
                            st.session_state.y_test = y_test
                            st.session_state.label_encoders = label_encoders
                            
                            # Retrain models
                            global models, scores_df
                            models, scores_df = train_models(X_train, y_train, X_test, y_test)
                            
                            # Clean up
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                                
                            st.success("System updated successfully! All pages will now use the new dataset.")
                            st.experimental_rerun()
                            
                        except ValueError as ve:
                            st.error(f"Validation Error: {str(ve)}")
                        except Exception as e:
                            st.error(f"Error processing dataset: {str(e)}")
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    elif password:
        st.error("Incorrect password")

# --- Help ---
elif page == "Help":
    st.title("User Manual")
    st.write("""
    **Instructions:**
    - **Home:** Overview and dataset preview.
    - **Data Analysis:** Visualizations and model performance.
    - **Prediction:** Try predictions by selecting input values.
    - **Reports:** View dataset summary statistics.
    - **Admin:** Upload new datasets (admin only).
    
    **Note:** All uploaded datasets must contain an 'Injury Severity' column.
    """)
