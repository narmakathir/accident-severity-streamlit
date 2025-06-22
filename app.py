# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configure matplotlib dark mode
plt.style.use('dark_background')
params = {
    'axes.facecolor': '#0E1117',
    'figure.facecolor': '#0E1117',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white'
}
plt.rcParams.update(params)

# Try to import folium with fallback
try:
    import folium
    from streamlit_folium import folium_static
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

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
DARK_PALETTE = sns.color_palette("husl")
COLOR_MAP = "viridis"

# --- Project Overview ---
PROJECT_OVERVIEW = """
Traffic accidents are a major problem worldwide, causing several fatalities, damage to property, and loss of productivity. 
Predicting accident severity based on contributors such as weather conditions, road conditions, types of vehicles, and drivers 
enables the authorities to take necessary actions to minimize the risk and develop better emergency responses.
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
        'Surface Condition': {
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
        df = pd.read_csv(uploaded_file)
    
    # Validate required columns
    required_cols = ['Injury Severity', 'Location']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing in dataset")

    # Clean data
    df.drop_duplicates(inplace=True)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns
    
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Extract coordinates
    if 'Location' in df.columns:
        location = df['Location'].str.replace(r'[()]', '', regex=True).str.split(',', expand=True)
        df['latitude'] = pd.to_numeric(location[0], errors='coerce')
        df['longitude'] = pd.to_numeric(location[1], errors='coerce')
        df.dropna(subset=['latitude', 'longitude'], inplace=True)
    
    df = normalize_categories(df)

    # Label encoding
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col != 'Location':
            df[col] = df[col].astype(str).str.strip().str.title()
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Standard scaling
    target_col = 'Injury Severity'
    numeric_cols = df.select_dtypes(include=np.number).columns.difference([target_col])
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Train-test split
    X = df.drop([target_col, 'Location'], axis=1, errors='ignore')
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return df, X, y, X_train, X_test, y_train, y_test, label_encoders

# Initialize session state
if 'df' not in st.session_state:
    try:
        df, X, y, X_train, X_test, y_train, y_test, label_encoders = load_data()
        st.session_state.update({
            'df': df, 'X': X, 'y': y,
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'label_encoders': label_encoders
        })
    except Exception as e:
        st.error(f"Initial data loading failed: {str(e)}")

# --- Train Models ---
@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'Artificial Neural Network': MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=300,
            activation='relu',
            solver='adam',
            random_state=42
        )
    }
    
    trained_models = {}
    model_scores = []
    importances_dict = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Store feature importances
        if name == 'Logistic Regression':
            importance = np.abs(model.coef_[0])
        elif name == 'Artificial Neural Network':
            importance = np.mean(np.abs(model.coefs_[0]), axis=1)
        else:
            importance = model.feature_importances_
        
        importances_dict[name] = importance / np.sum(importance)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        trained_models[name] = model
        model_scores.append([name, acc*100, prec*100, rec*100, f1*100])

    scores_df = pd.DataFrame(model_scores, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
    return trained_models, scores_df, importances_dict

if 'models' not in st.session_state and 'X_train' in st.session_state:
    try:
        models, scores_df, importances_dict = train_models(
            st.session_state.X_train, st.session_state.y_train,
            st.session_state.X_test, st.session_state.y_test
        )
        st.session_state.update({
            'models': models,
            'scores_df': scores_df,
            'importances_dict': importances_dict
        })
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")

# --- Admin Upload ---
def admin_upload():
    st.title("Admin Dataset Upload")
    st.warning("This section is for administrators only.")
    
    password = st.text_input("Enter Admin Password", type="password")
    if password != "admin123":
        st.error("Incorrect password")
        return
    
    uploaded_file = st.file_uploader("Upload new dataset (CSV)", type="csv")
    if uploaded_file is not None:
        try:
            with st.spinner("Processing new dataset..."):
                # Clear all cached data
                st.cache_data.clear()
                st.cache_resource.clear()
                
                # Load new data
                df, X, y, X_train, X_test, y_train, y_test, label_encoders = load_data(uploaded_file)
                
                # Update session state
                st.session_state.update({
                    'df': df, 'X': X, 'y': y,
                    'X_train': X_train, 'X_test': X_test,
                    'y_train': y_train, 'y_test': y_test,
                    'label_encoders': label_encoders
                })
                
                # Retrain models
                models, scores_df, importances_dict = train_models(X_train, y_train, X_test, y_test)
                st.session_state.update({
                    'models': models,
                    'scores_df': scores_df,
                    'importances_dict': importances_dict
                })
                
            st.success("Dataset updated successfully!")
            st.experimental_rerun()
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Please ensure the file contains all required columns including 'Injury Severity'")

# --- Visualization Functions ---
def plot_severity_distribution():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Injury Severity', data=st.session_state.df, ax=ax, palette=DARK_PALETTE)
    ax.set_title('Distribution of Injury Severity', pad=20, color='white')
    ax.set_xlabel('Injury Severity Level', labelpad=10, color='white')
    ax.set_ylabel('Number of Accidents', labelpad=10, color='white')
    ax.tick_params(axis='both', colors='white')
    st.pyplot(fig)

def plot_correlation_heatmap():
    corr_cols = [
        'Driver At Fault', 'Driver Distracted By', 'Vehicle Damage Extent',
        'Traffic Control', 'Weather', 'Surface Condition', 'Light',
        'Speed Limit', 'Driver Substance Abuse', 'Injury Severity'
    ]
    corr_df = st.session_state.df[[c for c in corr_cols if c in st.session_state.df.columns]]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_df.corr(), cmap=COLOR_MAP, annot=False, ax=ax)
    ax.set_title("Feature Correlation Heatmap", pad=20, color='white')
    st.pyplot(fig)

def plot_model_comparison():
    fig, ax = plt.subplots(figsize=(12, 6))
    st.session_state.scores_df.set_index('Model').plot(
        kind='bar', ax=ax, color=DARK_PALETTE
    )
    ax.set_title('Model Performance Comparison', pad=20, color='white')
    ax.set_ylabel('Score (%)', labelpad=10, color='white')
    ax.set_xlabel('Model', labelpad=10, color='white')
    ax.tick_params(axis='x', rotation=45, colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        facecolor='#0E1117',
        edgecolor='white',
        labelcolor='white'
    )
    st.pyplot(fig)

def plot_feature_importance(model_name):
    importance = st.session_state.importances_dict[model_name]
    sorted_idx = np.argsort(importance)[::-1][:10]
    top_features = st.session_state.X.columns[sorted_idx]
    top_importance = importance[sorted_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_importance, y=top_features, ax=ax, palette=DARK_PALETTE)
    ax.set_title(f'{model_name} - Feature Importance', pad=20, color='white')
    ax.set_xlabel('Importance Score', labelpad=10, color='white')
    ax.set_ylabel('Feature', labelpad=10, color='white')
    ax.tick_params(colors='white')
    st.pyplot(fig)

# --- Side Menu ---
st.sidebar.title("Navigation")
pages = ["Home", "Data Analysis", "Prediction", "Reports", "Admin Upload"]
page = st.sidebar.radio("Go to", pages)

# --- Page Routing ---
if page == "Home":
    st.title("Traffic Accident Severity Prediction")
    st.write(PROJECT_OVERVIEW)
    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.df.head() if 'df' in st.session_state else pd.DataFrame())

elif page == "Data Analysis":
    st.title("Data Analysis & Insights")
    
    if 'df' not in st.session_state:
        st.error("Data not loaded. Please upload dataset in Admin section.")
        return
    
    st.subheader("Injury Severity Distribution")
    plot_severity_distribution()
    st.divider()
    
    st.subheader("Accident Hotspots")
    if 'latitude' in st.session_state.df.columns:
        if FOLIUM_AVAILABLE:
            try:
                m = folium.Map(
                    location=[st.session_state.df['latitude'].mean(), 
                             st.session_state.df['longitude'].mean()],
                    zoom_start=11,
                    tiles='CartoDB dark_matter'
                )
                for _, row in st.session_state.df.sample(1000).iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=3,
                        color='red',
                        fill=True,
                        fill_color='red',
                        fill_opacity=0.7
                    ).add_to(m)
                folium_static(m, width=1000, height=500)
            except Exception as e:
                st.error(f"Map error: {str(e)}")
                st.map(st.session_state.df[['latitude', 'longitude']].dropna())
        else:
            st.map(st.session_state.df[['latitude', 'longitude']].dropna())
    else:
        st.warning("Location data not available")
    st.divider()
    
    st.subheader("Feature Correlations")
    plot_correlation_heatmap()
    st.divider()
    
    if 'models' in st.session_state:
        st.subheader("Model Performance")
        st.dataframe(st.session_state.scores_df.style.format("{:.2f}"))
        st.divider()
        
        st.subheader("Model Comparison")
        plot_model_comparison()
        st.divider()
        
        st.subheader("Feature Importance")
        model_name = st.selectbox(
            "Select Model", 
            list(st.session_state.models.keys()),
            index=1
        )
        plot_feature_importance(model_name)
    else:
        st.warning("Models not trained. Please check data loading.")

elif page == "Prediction":
    st.title("Custom Prediction")
    
    if 'models' not in st.session_state:
        st.error("Models not available. Please check data loading.")
        return
    
    selected_model = st.selectbox(
        "Choose Model", 
        list(st.session_state.models.keys())
    )
    
    input_data = {}
    cols = st.columns(3)
    
    for i, col in enumerate(st.session_state.X.columns):
        current_col = cols[i % 3]
        with current_col:
            if col in st.session_state.label_encoders:
                options = st.session_state.label_encoders[col].classes_
                choice = st.selectbox(col, options)
                input_data[col] = st.session_state.label_encoders[col].transform([choice])[0]
            else:
                input_data[col] = st.number_input(
                    col,
                    min_value=float(st.session_state.df[col].min()),
                    max_value=float(st.session_state.df[col].max()),
                    value=float(st.session_state.df[col].median())
                )
    
    if st.button("Predict Severity"):
        input_df = pd.DataFrame([input_data])
        model = st.session_state.models[selected_model]
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        confidence = np.max(proba) * 100
        
        if 'Injury Severity' in st.session_state.label_encoders:
            severity = st.session_state.label_encoders['Injury Severity'].inverse_transform([prediction])[0]
        else:
            severity = prediction
        
        st.success(f"**Predicted Injury Severity:** {severity}")
        st.info(f"**Confidence:** {confidence:.2f}%")

elif page == "Reports":
    st.title("Dataset Reports")
    if 'df' in st.session_state:
        st.subheader("Statistical Summary")
        st.dataframe(st.session_state.df.describe())
        
        st.subheader("Missing Values Check")
        st.write(st.session_state.df.isnull().sum())
    else:
        st.error("Data not loaded")

elif page == "Admin Upload":
    admin_upload()
