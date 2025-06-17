# --- Imports --- 
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

# --- Load Dataset --- 
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/narmakathir/accident-severity-streamlit/main/filtered_crash_data.csv'
    df = pd.read_csv(url)
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col != 'Location':
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
    model_scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        trained_models[name] = model
        model_scores[name] = [acc, prec, rec, f1]

    return trained_models, model_scores

models, model_scores = train_models()

# --- Page Navigation --- 
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Custom Prediction Interface", "Reports", "User Manual", "Admin Page"])

# --- Home --- 
if page == "Home":
    st.title("Traffic Accident Severity Prediction")
    st.write("""
    **Project Overview:**  
    Traffic accidents are a major problem worldwide, causing numerous fatalities, damage to property, and productivity losses.  
    Predicting accident severity based on contributors such as road conditions, vehicle types, and driver behavior can help authorities respond faster and implement preventive measures.  
    This project uses Machine Learning techniques to analyze past accidents and predict their severity, thereby helping to improve road safety.
    """)

    st.write("### Dataset Preview")
    st.dataframe(df.copy().head(10))

    st.write("### Dataset Summary")
    st.write(f"- Number of Rows: {len(df)}")
    st.write(f"- Number of Columns: {len(df.columns)}")
    st.write("- Columns:")
    for col in df.columns:
        st.write(f"  - {col}")

# --- Data Analysis --- 
elif page == "Data Analysis":
    st.title("Data Analysis")
    st.write("### Distribution of Injury Severity")
    fig, ax = plt.subplots()
    sns.countplot(x='Injury Severity', data=df, ax=ax)
    ax.set_title("Count of Injury Severity")
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax)
    st.pyplot(fig)

    st.write("### Hotspot Map")
    if 'Location' in df.columns:
        df_location = df.copy()
        coords = df_location['Location'].str.split(',', expand=True)
        df_location['latitude'] = coords[0].apply(float)
        df_location['longitude'] = coords[1].apply(float)

        st.map(df_location[['latitude', 'longitude']].dropna()) 
    else:
        st.error("Location data not available.")
    
    st.write("### Model Performance")
    performance_df = pd.DataFrame(model_scores, index=['Accuracy', 'Precision', 'Recall', 'F1-Score']).T * 100
    st.dataframe(performance_df.style.background_gradient(cmap='YlGn'))
  
    st.write("### Model Comparison")
    fig, ax = plt.subplots()
    performance_df.plot(kind='bar', ax=ax)
    ax.set_title('Model Comparison')
    ax.set_ylabel('Percentage')
    st.pyplot(fig)

    st.write("### Model Importances")
    model_name = st.selectbox("Select Model", list(models.keys()))

    # Prepare importances
    importances = []
    if hasattr(models[model_name], "feature_importances_"):
        importances = models[model_name].feature_importances_
    elif hasattr(models[model_name], "coef_"):
        importances = np.abs(models[model_name].coef_[0])

    if len(importances) > 0:
        df_imp = pd.DataFrame({"Feature": X.columns, "Importance": importances})
        df_imp = df_imp.sort_values(by='Importance', ascending=False)

        st.bar_chart(df_imp.set_index("Feature"))
    else:
        st.info("Feature importances not available for this model.")
    

# --- Custom Prediction Interface --- 
elif page == "Custom Prediction Interface":
    st.title("Custom Prediction")
    selected_model = st.selectbox("Choose Model for Prediction", list(models.keys()))
    model = models[selected_model]

    input_data = {}
    for col in X.columns:
        if col in label_encoders:
            options = list(label_encoders[col].classes_)
            choice = st.selectbox(f"{col}", options)
            input_data[col] = label_encoders[col].transform([choice])[0]
        else:
            input_data[col] = st.number_input(f"{col}", float(np.min(X[col])), float(np.max(X[col])), float(np.mean(X[col])))

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    confidence = np.max(probs) * 100

    severity_label = label_encoders['Injury Severity'].inverse_transform([prediction])[0] if 'Injury Severity' in label_encoders else prediction
    st.success(f"**Predicted Injury Severity:** {severity_label}")
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
    - **Home:** View Project Overview and Summary.
    - **Data Analysis:** Distribution, Hotspots, Model Performance, Importances.
    - **Custom Prediction Interface:** Input custom data to predict severity.
    - **Reports:** Summary reports for stakeholders.
    - **Admin:** (For administrators) Update or manage the dataset.
    """)

# --- Admin --- 
elif page == "Admin Page":
    st.title("Admin Panel")
    uploaded_file = st.file_uploader("Upload new dataset", type=["csv"])

    if uploaded_file:
        st.success("Dataset uploaded. Reload the application to apply.")
