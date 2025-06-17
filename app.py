# --- Imports --- 
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# --- Config --- 
st.set_page_config(page_title="Accident Severity Predictor", layout="wide")

# --- Prepare Data --- 
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/narmakathir/accident-severity-streamlit/main/filtered_crash_data.csv'
    df = pd.read_csv(url)
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    # Split Location into latitude and longitude
    df[['latitude', 'longitude']] = df['Location'].str.extract(r'\((.*),(.*)\)')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col not in ['Location']:
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

df, X, y, X_train, X_test, y_train, y_test, label_encoders = load_data()

# --- Train Models --- 
@st.cache_resource
def train_models():
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
        model_scores.append([name, acc, prec, rec, f1])

    scores_df = pd.DataFrame(model_scores, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
    return trained_models, scores_df

models, scores_df = train_models()

# --- Sidebar --- 
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Predictions", "Custom Prediction Interface", "Reports", "User Manual", "Admin Page"])

# --- Home --- 
if page == "Home":
    st.title("Predicting Traffic Accident Severity Using Machine Learning")
    st.write("### Project Overview")
    st.write("""
    Traffic accidents are a major problem worldwide, causing several fatalities, damage to property, and loss of productivity. 
    Predicting accident severity based on contributors such as weather conditions, road conditions, vehicle types, and drivers enables authorities to take necessary actions to minimize the risk and develop better emergency responses. 
    This project uses machine learning techniques to analyze past traffic data for accident severity prediction and present useful data to improve road safety and management.
    """)

    st.write("### Dataset Summary")
    st.write(f"**Total cases:** {len(df)}")
    breakdown = (df['Injury Severity'].value_counts(normalize=True)*100).reset_index()
    breakdown.columns = ['Class', 'Percentage']

    st.bar_chart(breakdown.set_index('Class'))
    st.write(breakdown)

# --- Data Analysis --- 
elif page == "Data Analysis":
    st.title("Data Analysis")
    st.write("### Countplot of Injury Severity")
    fig, ax = plt.subplots()
    sns.countplot(x='Injury Severity', data=df, palette='Blues')
    ax.set_title('Distribution of Injury Severity')
    st.pyplot(fig)

    st.write("### Heatmap of Pearson's Correlations")
    fig, ax = plt.subplots()
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

    st.write("### Model Performance")
    st.dataframe(scores_df.style.background_gradient(cmap='YlGn'))  

    st.write("### Model Comparison")
    fig, ax = plt.subplots()
    scores_df.plot(x='Model', kind='bar', ax=ax)
    ax.set_title('Model Comparison')
    ax.set_ylabel('Score')
    st.pyplot(fig)

    st.write("### Hotspot Map")
    if 'latitude' in df.columns and 'longitude' in df.columns:
        st.map(df[['latitude', 'longitude']].dropna()) 
    else:
        st.error("Location data unavailable.")
    

# --- Predictions --- 
elif page == "Predictions":
    st.title("Make Predictions from Dataset")
    selected_model = st.selectbox("Choose Model", list(models.keys()))
    model = models[selected_model]
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(f"### {selected_model} - Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())    

# --- Custom Prediction --- 
elif page == "Custom Prediction Interface":
    st.title("Custom Prediction Interface")
    selected_model = st.selectbox("Choose Model for Prediction", list(models.keys()))
    model = models[selected_model]

    input_data = {}
    for col in X.columns:
        if col in label_encoders:
            options = list(label_encoders[col].classes_)
            choice = st.selectbox(f"{col}", options)
            input_data[col] = label_encoders[col].transform([choice])[0]
        else:
            input_data[col] = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    confidence = np.max(probs) * 100

    severity_label = label_encoders['Injury Severity'].inverse_transform([prediction])[0] if 'Injury Severity' in label_encoders else prediction
    st.success(f"**Predicted Severity:** {severity_label}")
    st.info(f"**Confidence:** {confidence:.2f}%")

# --- Reports --- 
elif page == "Reports":
    st.title("Generated Reports")
    st.write("### Dataset Summary")
    st.dataframe(df.describe())    

# --- User Manual --- 
elif page == "User Manual":
    st.title("User Manual")
    st.write("""
    **Instructions:**  
    - **Data Analysis:** View all charts, correlations, and model performance.  
    - **Predictions:** Validate models against test set.  
    - **Custom Prediction:** Provide custom input to predict accident severity.  
    - **Reports:** View summary reports.  
    """)

# --- Admin --- 
elif page == "Admin Page":
    st.title("Admin Panel")
    uploaded_file = st.file_uploader("Upload new dataset", type=["csv"])
    if uploaded_file:
        st.success("Dataset uploaded. Reload app to use new data.")
