# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
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
sns.set_style("darkgrid")
plt.style.use('dark_background')
PALETTE = sns.color_palette("Set2")

# Set dark background for all visualizations
plt.rcParams['figure.facecolor'] = '#0E1117'
plt.rcParams['axes.facecolor'] = '#0E1117'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['text.color'] = 'white'

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
def load_data(url=None):
    if url is None:
        url = 'https://raw.githubusercontent.com/narmakathir/accident-severity-streamlit/main/filtered_crash_data.csv'
    
    if os.path.exists(url):
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(url)
        
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    df['Location_Original'] = df['Location']  # Preserve original for mapping
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

    X = df.drop([target_col, 'Location'], axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return df, X, y, X_train, X_test, y_train, y_test, label_encoders

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

# --- Session State Management ---
if 'data_loaded' not in st.session_state:
    df, X, y, X_train, X_test, y_train, y_test, label_encoders = load_data()
    models, scores_df = train_models(X_train, y_train, X_test, y_test)
    st.session_state.update({
        'df': df,
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoders': label_encoders,
        'models': models,
        'scores_df': scores_df,
        'data_loaded': True
    })

# --- Side Menu ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Prediction", "Reports", "Admin", "Help"])

# Helper function to get session state variables
def get_state():
    return (
        st.session_state.df,
        st.session_state.X,
        st.session_state.y,
        st.session_state.X_train,
        st.session_state.X_test,
        st.session_state.y_train,
        st.session_state.y_test,
        st.session_state.label_encoders,
        st.session_state.models,
        st.session_state.scores_df
    )

# --- Home ---
if page == "Home":
    st.title("Traffic Accident Severity Prediction")
    st.write(PROJECT_OVERVIEW)
    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.df.copy().head())

# --- Data Analysis ---
elif page == "Data Analysis":
    df, X, y, _, _, _, _, label_encoders, models, scores_df = get_state()
    
    st.title("Data Analysis & Insights")
    st.markdown("*Explore key patterns and model performance.*")
    st.divider()

    st.subheader("➥ Injury Severity Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Injury Severity', data=df, ax=ax, palette=PALETTE)
    ax.set_title('Count of Injury Levels', color='white')
    ax.set_xlabel('Injury Severity', color='white')
    ax.set_ylabel('Count', color='white')
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
            st.warning("No geographic coordinates found in location data. Please check the format.")
    else:
        st.warning("Location data not found in the dataset.")
    st.divider()

    st.subheader("➥ Correlation Heatmap")
    corr = df.select_dtypes(['number']).corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", ax=ax, 
                annot_kws={"size": 8}, cbar_kws={"label": "Correlation Coefficient"})
    ax.set_title("Correlation Heatmap", pad=20)
    st.pyplot(fig)
    st.divider()

    st.subheader("➥ Model Performance")
    st.table(scores_df.round(2))
    st.divider()

    st.subheader("➥ Model Comparison Bar Chart")
    performance_df = scores_df.set_index('Model')
    fig, ax = plt.subplots(figsize=(10, 6))
    performance_df.plot(kind='bar', ax=ax, color=PALETTE.as_hex())
    ax.set_title('Model Comparison', pad=20)
    ax.set_ylabel('Score (%)')
    ax.set_xlabel('Model')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
    top_features = X.columns[sorted_idx][:10]
    top_vals = importances_vals[sorted_idx][:10]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_vals, y=top_features, ax=ax, palette=PALETTE)
    ax.set_title(f'{model_name} Top 10 Features', pad=20)
    ax.set_xlabel('Importance Score', color='white')
    ax.set_ylabel('Features', color='white')
    st.pyplot(fig)

# --- Prediction ---
elif page == "Prediction":
    df, X, y, _, _, _, _, label_encoders, models, _ = get_state()
    
    st.title("Custom Prediction")
    selected_model = st.selectbox("Choose Model for Prediction", list(models.keys()))
    model = models[selected_model]

    input_data = {}
    cols = st.columns(2)
    for i, col in enumerate(X.columns):
        with cols[i % 2]:
            if col in label_encoders:
                options = sorted(label_encoders[col].classes_)
                choice = st.selectbox(f"{col}", options)
                input_data[col] = label_encoders[col].transform([choice])[0]
            else:
                input_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    if st.button("Predict Severity"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        confidence = np.max(probs) * 100

        if 'Injury Severity' in label_encoders:
            severity_label = label_encoders['Injury Severity'].inverse_transform([prediction])[0]
        else:
            severity_label = prediction

        st.success(f"**Predicted Injury Severity:** {severity_label}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        # Show probability distribution
        fig, ax = plt.subplots()
        if 'Injury Severity' in label_encoders:
            classes = label_encoders['Injury Severity'].classes_
        else:
            classes = range(len(probs))
            
        sns.barplot(x=classes, y=probs, ax=ax, palette=PALETTE)
        ax.set_title('Probability Distribution')
        ax.set_xlabel('Severity Level')
        ax.set_ylabel('Probability')
        st.pyplot(fig)

# --- Reports ---
elif page == "Reports":
    df, _, _, _, _, _, _, _, _, _ = get_state()
    st.title("Generated Reports")
    st.write("### Dataset Summary")
    st.dataframe(df.describe())

# --- Admin Page ---
elif page == "Admin":
    st.title("Admin Dashboard")
    st.subheader("Upload New Dataset")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Save the uploaded file
            with open("uploaded_dataset.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("Update System with New Data"):
                with st.spinner("Processing new data and retraining models..."):
                    # Clear cache to force reload
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    
                    # Reload data with the new file
                    new_df, new_X, new_y, new_X_train, new_X_test, new_y_train, new_y_test, new_label_encoders = load_data("uploaded_dataset.csv")
                    
                    # Retrain models with new data
                    new_models, new_scores_df = train_models(new_X_train, new_y_train, new_X_test, new_y_test)
                    
                    # Update session state
                    st.session_state.update({
                        'df': new_df,
                        'X': new_X,
                        'y': new_y,
                        'X_train': new_X_train,
                        'X_test': new_X_test,
                        'y_train': new_y_train,
                        'y_test': new_y_test,
                        'label_encoders': new_label_encoders,
                        'models': new_models,
                        'scores_df': new_scores_df
                    })
                    
                    st.success("System updated successfully with new data!")
                    st.balloons()
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# --- Help ---
elif page == "Help":
    st.title("User Manual")
    st.write("""
    **Instructions:**
    - **Home:** Overview and dataset preview.
    - **Data Analysis:** Visualizations and model performance.
    - **Prediction:** Try predictions by selecting input values.
    - **Reports:** View dataset summary statistics.
    - **Admin:** Upload new datasets to update the system.
    
    **Dark Mode:** All visualizations now use dark mode for better readability.
    
    **Geographic Data:** If location data isn't displaying, ensure your dataset contains coordinates in (latitude, longitude) format.
    """)
