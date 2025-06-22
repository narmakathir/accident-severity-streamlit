# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from datetime import datetime

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

# --- Session State Initialization ---
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'target_col' not in st.session_state:
    st.session_state.target_col = None

# --- Project Overview ---
PROJECT_OVERVIEW = """
Traffic accidents are a major problem worldwide, causing several fatalities, damage to property, and loss of productivity. Predicting accident severity based on contributors such as weather conditions, road conditions, types of vehicles, and drivers enables the authorities to take necessary actions to minimize the risk and develop better emergency responses. 
 
This project uses machine learning techniques to analyze past traffic data for accident severity prediction and present useful data to improve road safety and management.
"""

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
def load_data(uploaded_file=None, target_column=None):
    if uploaded_file is None:
        # Load default dataset
        url = 'https://raw.githubusercontent.com/narmakathir/accident-severity-streamlit/main/filtered_crash_data.csv'
        df = pd.read_csv(url)
    else:
        # Load uploaded file
        try:
            df = pd.read_csv(uploaded_file)
        except:
            try:
                df = pd.read_excel(uploaded_file)
            except:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return None, None, None, None, None, None, None, None, None
    
    # Clean data
    df = df.copy()
    df.drop_duplicates(inplace=True)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Fill remaining missing values with mode
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Preserve original location if exists
    if 'Location' in df.columns:
        df['Location_Original'] = df['Location']
    
    # Normalize categories
    df = normalize_categories(df)
    
    # If target column not specified, try to guess
    if target_column is None:
        possible_targets = ['severity', 'injury', 'accident', 'target', 'class']
        for col in df.columns:
            if any(word in col.lower() for word in possible_targets):
                target_column = col
                break
    
    # If still no target, use last column
    if target_column is None:
        target_column = df.columns[-1]
    
    st.session_state.target_col = target_column
    
    # Label encoding for categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.title()
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    st.session_state.label_encoders = label_encoders
    
    # Standard scaling for numeric features
    numeric_cols = df.select_dtypes(include='number').columns.difference([target_column])
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Prepare features and target
    X = df.drop([target_column, 'Location'] if 'Location' in df.columns else [target_column], axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.session_state.current_df = df
    st.session_state.data_loaded = True
    
    return df, X, y, X_train, X_test, y_train, y_test, label_encoders, target_column

# --- Train Models ---
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

# --- Initialize Data ---
if not st.session_state.data_loaded:
    df, X, y, X_train, X_test, y_train, y_test, label_encoders, target_col = load_data()
else:
    df = st.session_state.current_df
    label_encoders = st.session_state.label_encoders
    target_col = st.session_state.target_col
    X = df.drop([target_col, 'Location'] if 'Location' in df.columns else [target_col], axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models, scores_df = train_models(X_train, y_train, X_test, y_test)

# --- Side Menu ---
st.sidebar.title("Navigation")
if 'admin_mode' not in st.session_state:
    st.session_state.admin_mode = False

if st.sidebar.checkbox("Admin Mode"):
    st.session_state.admin_mode = True
else:
    st.session_state.admin_mode = False

if st.session_state.admin_mode:
    page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Prediction", "Reports", "Help", "Admin"])
else:
    page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Prediction", "Reports", "Help"])

# --- Home ---
if page == "Home":
    st.title("Traffic Accident Severity Prediction")
    st.write(PROJECT_OVERVIEW)

    st.subheader("Dataset Preview")
    st.dataframe(df.copy().head())

# --- Data Analysis ---
elif page == "Data Analysis":
    st.title("Data Analysis & Insights")
    st.markdown("*Explore key patterns and model performance.*")
    st.divider()

    st.subheader("➥ Target Variable Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=target_col, data=df, ax=ax, palette=PALETTE)
    ax.set_title(f'Count of {target_col} Levels')
    st.pyplot(fig)
    st.divider()

    st.subheader("➥ Hotspot Location")
    if 'Location_Original' in df.columns:
        coords = df['Location_Original'].astype(str).str.extract(r'\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)')
        coords.columns = ['latitude', 'longitude']
        coords = coords.astype(float).dropna()
        if not coords.empty:
            st.map(coords)
        else:
            st.warning("No geographic data available.")
    else:
        st.warning("Location data not found.")
    st.divider()

    st.subheader("➥ Correlation Heatmap")
    corr = df.select_dtypes(['number']).corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, cmap='YlGnBu', annot=False, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    st.divider()

    st.subheader("➥ Model Performance")
    st.table(scores_df.round(2))
    st.divider()

    st.subheader("➥ Model Comparison Bar Chart")
    performance_df = scores_df.set_index('Model')
    fig, ax = plt.subplots()
    performance_df.plot(kind='bar', ax=ax, color=PALETTE.as_hex())
    ax.set_title('Model Comparison')
    ax.set_ylabel('Score (%)')
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
    
    if model_name in importances:
        importances_vals = importances[model_name]
        importances_vals /= importances_vals.sum()

        sorted_idx = np.argsort(importances_vals)[::-1]
        top_features = X.columns[sorted_idx][:10]
        top_vals = importances_vals[sorted_idx][:10]

        fig, ax = plt.subplots()
        sns.barplot(x=top_vals, y=top_features, ax=ax, palette=PALETTE)
        ax.set_title(f'{model_name} Top 10 Features')
        st.pyplot(fig)
    else:
        st.warning(f"Feature importance not available for {model_name}")

# --- Prediction ---
elif page == "Prediction":
    st.title("Custom Prediction")
    selected_model = st.selectbox("Choose Model for Prediction", list(models.keys()))
    model = models[selected_model]

    input_data = {}
    for col in X.columns:
        if col in label_encoders:
            options = sorted(label_encoders[col].classes_)
            choice = st.selectbox(f"{col}", options)
            input_data[col] = label_encoders[col].transform([choice])[0]
        else:
            input_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

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

# --- Reports ---
elif page == "Reports":
    st.title("Generated Reports")
    st.write("### Dataset Summary")
    st.dataframe(df.describe())

# --- Help ---
elif page == "Help":
    st.title("User Manual")
    st.write("""
    **Instructions:**
    - **Home:** Overview and dataset preview.
    - **Data Analysis:** Visualizations and model performance.
    - **Prediction:** Try predictions by selecting input values.
    - **Reports:** View dataset summary statistics.
    - **Admin:** (Admin only) Upload new datasets and manage system.
    """)

# --- Admin Page ---
elif page == "Admin":
    st.title("Admin Dashboard")
    st.warning("This page is for administrators only. Changes here will affect all users.")
    
    st.subheader("Upload New Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    st.subheader("Dataset Configuration")
    if uploaded_file is not None:
        # Preview the uploaded file
        try:
            preview_df = pd.read_csv(uploaded_file)
        except:
            try:
                preview_df = pd.read_excel(uploaded_file)
            except:
                st.error("Could not read the file. Please check the format.")
        
        st.write("File Preview:")
        st.dataframe(preview_df.head())
        
        # Let admin select target column
        target_column = st.selectbox(
            "Select the target column for prediction",
            options=preview_df.columns,
            index=len(preview_df.columns)-1
        )
        
        # Let admin specify custom category mappings
        st.subheader("Category Normalization")
        st.write("Specify how to normalize categorical values (optional)")
        
        categorical_cols = preview_df.select_dtypes(include='object').columns
        custom_mappings = {}
        
        for col in categorical_cols:
            unique_values = preview_df[col].astype(str).unique()
            if len(unique_values) > 10:
                continue  # Skip columns with too many unique values
                
            st.write(f"**{col}**")
            cols = st.columns(2)
            with cols[0]:
                st.write("Original values:")
                st.write(unique_values)
            with cols[1]:
                st.write("Replace with:")
                replacements = {}
                for val in unique_values:
                    replacements[val] = st.text_input(f"Replace '{val}' with:", val, key=f"{col}_{val}")
            custom_mappings[col] = replacements
        
        if st.button("Load New Dataset"):
            with st.spinner("Processing new dataset..."):
                # Reset session state
                st.session_state.data_loaded = False
                st.session_state.current_df = None
                st.session_state.label_encoders = {}
                
                # Reload data with new file
                uploaded_file.seek(0)  # Reset file pointer
                df, X, y, X_train, X_test, y_train, y_test, label_encoders, target_col = load_data(
                    uploaded_file=uploaded_file,
                    target_column=target_column
                )
                
                # Retrain models
                models, scores_df = train_models(X_train, y_train, X_test, y_test)
                
                st.success("Dataset successfully updated! All pages will now use the new data.")
                st.balloons()
    
    st.subheader("System Information")
    st.write(f"Current dataset shape: {df.shape if st.session_state.data_loaded else 'Not loaded'}")
    st.write(f"Target variable: {target_col if st.session_state.data_loaded else 'Not set'}")
    st.write(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if st.button("Reset to Default Dataset"):
        with st.spinner("Resetting to default dataset..."):
            st.session_state.data_loaded = False
            df, X, y, X_train, X_test, y_train, y_test, label_encoders, target_col = load_data()
            models, scores_df = train_models(X_train, y_train, X_test, y_test)
            st.success("Successfully reset to default dataset!")
