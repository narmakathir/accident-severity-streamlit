# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
# Set dark theme for all visualizations
plt.style.use('dark_background')
sns.set_style("darkgrid")
PALETTE = sns.color_palette("husl")

# --- Project Overview ---
PROJECT_OVERVIEW = """
Traffic accidents are a major problem worldwide, causing several fatalities, damage to property, and loss of productivity. 
Predicting accident severity based on contributors such as weather conditions, road conditions, types of vehicles, and drivers 
enables the authorities to take necessary actions to minimize the risk and develop better emergency responses. 
 
This project uses machine learning techniques to analyze past traffic data for accident severity prediction and present useful 
data to improve road safety and management.
"""

# --- Normalize Text Values ---
def normalize_categories(df):
    mappings = {
        'Weather': {
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
        'Light': {
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
def load_data(uploaded_file=None):
    if uploaded_file is None:
        url = 'https://raw.githubusercontent.com/narmakathir/accident-severity-streamlit/main/filtered_crash_data.csv'
        df = pd.read_csv(url)
    else:
        try:
            # First check if file is empty
            if uploaded_file.size == 0:
                st.error("Uploaded file is empty")
                st.stop()
            
            # Try reading the file
            df = pd.read_csv(uploaded_file)
            
            # Check if dataframe is empty
            if df.empty:
                st.error("Uploaded file contains no data")
                st.stop()
                
            # Check if dataframe has no columns
            if len(df.columns) == 0:
                st.error("Uploaded file has no recognizable columns")
                st.stop()
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.stop()
    
    # Basic data cleaning
    df.drop_duplicates(inplace=True)
    
    # Handle missing values - numeric first, then categorical
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include='object').columns
    if len(categorical_cols) > 0:
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    # Extract coordinates from Location if available
    if 'Location' in df.columns:
        try:
            coords = df['Location'].str.extract(r'\(([^,]+),\s*([^)]+)\)')
            if not coords.empty:
                df['latitude'] = pd.to_numeric(coords[0], errors='coerce')
                df['longitude'] = pd.to_numeric(coords[1], errors='coerce')
                df.dropna(subset=['latitude', 'longitude'], inplace=True)
        except Exception as e:
            st.warning(f"Could not parse location data: {str(e)}")

    df = normalize_categories(df)

    # Label encoding for categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.title()
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Handle target column - flexible naming
    target_col = None
    possible_targets = ['Injury Severity', 'Severity', 'Injury', 'Accident Severity', 'Target', 'Accident_Severity']
    for possible_target in possible_targets:
        if possible_target in df.columns:
            target_col = possible_target
            break
    
    if target_col is None:
        st.warning("⚠️ Could not find standard target column. Using first numeric column as target.")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[0]
        else:
            st.error("No suitable target column found in the dataset")
            st.stop()

    # Feature scaling
    numeric_cols = df.select_dtypes(include=np.number).columns.difference([target_col])
    scaler = StandardScaler()
    if len(numeric_cols) > 0:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Prepare features - handle missing columns gracefully
    columns_to_drop = [target_col]
    if 'Location' in df.columns:
        columns_to_drop.append('Location')
    
    X = df.drop(columns=columns_to_drop, errors='ignore')
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return df, X, y, X_train, X_test, y_train, y_test, label_encoders, target_col

# Initialize session state for uploaded file
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

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

# Load data based on uploaded file or default
try:
    df, X, y, X_train, X_test, y_train, y_test, label_encoders, target_col = load_data(st.session_state.uploaded_file)
    models, scores_df = train_models(X_train, y_train, X_test, y_test)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# --- Side Menu ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Prediction", "Reports", "Admin"])

# --- Home ---
if page == "Home":
    st.title("Traffic Accident Severity Prediction")
    st.write(PROJECT_OVERVIEW)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")
    st.write(f"Total records: {len(df)}")
    st.write(f"Number of features: {len(df.columns)}")
    
    if target_col in df.columns:
        st.write(f"Target variable ('{target_col}') distribution:")
        st.write(df[target_col].value_counts())

# --- Data Analysis ---
elif page == "Data Analysis":
    st.title("Data Analysis & Insights")
    st.markdown("*Explore key patterns and model performance.*")
    st.divider()

    # Target Distribution
    st.subheader(f"➥ Target Variable Distribution")
    if target_col in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=target_col, data=df, ax=ax, palette=PALETTE)
        ax.set_title(f'Distribution of {target_col}', color='white')
        ax.set_xlabel(target_col, color='white')
        ax.set_ylabel('Count', color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)
    else:
        st.warning("⚠️ Target column not found - cannot display distribution")
    st.divider()

    # Hotspot Location Map
    st.subheader("➥ Hotspot Location")
    if all(col in df.columns for col in ['latitude', 'longitude']):
        try:
            center_lat = df['latitude'].mean()
            center_lon = df['longitude'].mean()
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='CartoDB dark_matter')
            
            sample_size = min(1000, len(df))
            sample_df = df.sample(n=sample_size, random_state=42)
            
            for _, row in sample_df.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.7
                ).add_to(m)
            
            folium_static(m)
            st.caption(f"Showing {sample_size} random accident locations")
        except Exception as e:
            st.error(f"Error creating map: {str(e)}")
    elif 'Location' in df.columns:
        st.warning("⚠️ Could not parse location coordinates from 'Location' column")
    else:
        st.info("ℹ️ Location data not available - map visualization requires 'latitude'/'longitude' or 'Location' columns")
    st.divider()

    # Correlation Heatmap
    st.subheader("➥ Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax)
        ax.set_title("Feature Correlation Heatmap", color='white')
        st.pyplot(fig)
    else:
        st.info("ℹ️ Not enough numeric columns for correlation analysis")
    st.divider()

    # Model Performance
    st.subheader("➥ Model Performance")
    st.dataframe(scores_df.style.format({
        'Accuracy (%)': '{:.2f}',
        'Precision (%)': '{:.2f}',
        'Recall (%)': '{:.2f}',
        'F1-Score (%)': '{:.2f}'
    }))
    st.divider()

    # Model Comparison Chart
    st.subheader("➥ Model Comparison")
    if not scores_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        scores_df.set_index('Model').plot(kind='bar', ax=ax, color=PALETTE)
        ax.set_title('Model Performance Comparison', color='white')
        ax.set_ylabel('Score (%)', color='white')
        ax.set_xlabel('Model', color='white')
        ax.tick_params(axis='x', rotation=45, colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.legend(facecolor='#0E1117', edgecolor='white')
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
    st.divider()

    # Feature Importance
    st.subheader("➥ Feature Importance")
    model_name = st.selectbox("Select Model", list(models.keys()), index=1)
    
    try:
        if model_name == 'Logistic Regression':
            importances = np.abs(models[model_name].coef_[0])
        elif model_name == 'Artificial Neural Network':
            importances = np.mean(np.abs(models[model_name].coefs_[0]), axis=1)
        else:  # Random Forest or XGBoost
            importances = models[model_name].feature_importances_
        
        importances = importances / importances.sum()  # Normalize
        sorted_idx = np.argsort(importances)[::-1]
        top_features = X.columns[sorted_idx][:10]
        top_importances = importances[sorted_idx][:10]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_importances, y=top_features, ax=ax, palette=PALETTE)
        ax.set_title(f'{model_name} - Top 10 Features', color='white')
        ax.set_xlabel('Relative Importance', color='white')
        ax.set_ylabel('Feature', color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"⚠️ Could not display feature importance: {str(e)}")

# --- Prediction ---
elif page == "Prediction":
    st.title("Custom Prediction")
    selected_model = st.selectbox("Choose Model for Prediction", list(models.keys()))
    model = models[selected_model]

    input_data = {}
    cols_per_row = 3
    cols = st.columns(cols_per_row)
    
    for i, col in enumerate(X.columns):
        with cols[i % cols_per_row]:
            if col in label_encoders:
                options = label_encoders[col].classes_
                choice = st.selectbox(f"{col}", options)
                input_data[col] = label_encoders[col].transform([choice])[0]
            else:
                input_data[col] = st.number_input(
                    f"{col}",
                    min_value=float(df[col].min()),
                    max_value=float(df[col].max()),
                    value=float(df[col].mean())
                )
    
    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            probs = model.predict_proba(input_df)[0]
            confidence = np.max(probs) * 100

            if target_col in label_encoders:
                severity_label = label_encoders[target_col].inverse_transform([prediction])[0]
            else:
                severity_label = prediction

            st.success(f"**Predicted {target_col}:** {severity_label}")
            st.info(f"**Confidence:** {confidence:.2f}%")
            
            # Show probability distribution
            fig, ax = plt.subplots(figsize=(8, 4))
            if target_col in label_encoders:
                classes = label_encoders[target_col].classes_
            else:
                classes = range(len(probs))
            
            sns.barplot(x=classes, y=probs, ax=ax, palette=PALETTE)
            ax.set_title('Prediction Probability Distribution', color='white')
            ax.set_xlabel('Severity Level', color='white')
            ax.set_ylabel('Probability', color='white')
            ax.tick_params(colors='white')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# --- Reports ---
elif page == "Reports":
    st.title("Dataset Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Statistics")
        st.dataframe(df.describe())
    
    with col2:
        st.subheader("Data Types")
        st.dataframe(df.dtypes.astype(str).to_frame('Data Type'))
    
    st.divider()
    
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.dataframe(missing_values[missing_values > 0].to_frame('Missing Values'))
    else:
        st.success("No missing values found in the dataset")
    
    st.divider()
    
    st.subheader("Column Information")
    col_info = []
    for col in df.columns:
        col_info.append({
            'Column': col,
            'Type': str(df[col].dtype),
            'Unique Values': df[col].nunique(),
            'Missing Values': df[col].isnull().sum()
        })
    st.dataframe(pd.DataFrame(col_info))

# --- Admin ---
elif page == "Admin":
    st.title("Admin Panel")
    st.warning("This section is for administrators only.")
    
    uploaded_file = st.file_uploader("Upload new dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            # First check if file is empty
            if uploaded_file.size == 0:
                st.error("⚠️ Uploaded file is empty")
                st.stop()
            
            # Try reading the first few bytes to check if it's a valid CSV
            content = uploaded_file.getvalue().decode('utf-8')
            if not content.strip():
                st.error("⚠️ Uploaded file contains no data")
                st.stop()
                
            # Try reading the file properly
            uploaded_file.seek(0)  # Reset file pointer
            test_df = pd.read_csv(uploaded_file)
            
            # Check if dataframe is empty
            if test_df.empty:
                st.error("⚠️ Uploaded file contains no data rows")
                st.stop()
                
            # Check if dataframe has no columns
            if len(test_df.columns) == 0:
                st.error("⚠️ Uploaded file has no recognizable columns")
                st.stop()
                
            # If we get here, file is valid
            st.session_state.uploaded_file = uploaded_file
            st.success("✅ New dataset uploaded successfully!")
            
            # Clear caches
            st.cache_data.clear()
            st.cache_resource.clear()
            
            st.info("ℹ️ Please refresh the page or navigate to another section to see updates")
            
        except pd.errors.EmptyDataError:
            st.error("⚠️ Uploaded file appears to be empty or corrupted")
        except UnicodeDecodeError:
            st.error("⚠️ File encoding issue - please upload a standard UTF-8 CSV file")
        except Exception as e:
            st.error(f"⚠️ Error reading uploaded file: {str(e)}")
    
    if st.button("Reset to Default Dataset"):
        st.session_state.uploaded_file = None
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Reset to default dataset complete! Refresh the page.")
