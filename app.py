import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import base64

# Set wide layout
st.set_page_config(layout="wide", page_title="Traffic Accident Severity Prediction")

@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1sVplp_5lFb3AMG5vWRqIltwNazLyM8vH"
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    drop_cols = ['Report Number', 'Local Case Number', 'Person ID', 'Vehicle ID',
                 'Latitude', 'Longitude', 'Location', 'Driverless Vehicle', 'Parked Vehicle']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    df.drop_duplicates(inplace=True)

    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    target_col = 'Injury Severity'
    numeric_cols = df.select_dtypes(include='number').columns.difference([target_col])
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, label_encoders, numeric_cols

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "accuracy": accuracy_score(y_test, y_pred) * 100,
        "precision": precision_score(y_test, y_pred, average='weighted') * 100,
        "recall": recall_score(y_test, y_pred, average='weighted') * 100,
        "f1_score": f1_score(y_test, y_pred, average='weighted') * 100,
    }

# ----------------- Streamlit UI --------------------

df = load_data()
df_clean, label_encoders, numeric_cols = preprocess_data(df)
target_col = 'Injury Severity'
accident_features = [
    'Driver At Fault', 'Circumstance', 'Driver Distracted By', 'Collision Type',
    'Vehicle Movement', 'Vehicle Going Dir', 'Vehicle First Impact Location',
    'Vehicle Damage Extent', 'Vehicle Body Type', 'Traffic Control',
    'Weather', 'Surface Condition', 'Light', 'Speed Limit', 'Driver Substance Abuse'
]
accident_features = [col for col in accident_features if col in df_clean.columns]

X = df_clean.drop(target_col, axis=1)
y = df_clean[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Dataset Details", "📈 Data Insights"])

# ---------- Home ----------
if page == "🏠 Home":
    st.title("Predicting Traffic Accident Severity Using Machine Learning")
    st.markdown("""
    This Streamlit app showcases a machine learning approach to predict the severity of road traffic accidents
    (Minor, Serious, Fatal) using real accident reports.
    
    **Objectives:**
    - Develop and evaluate ML models (RF, XGBoost, LR, ANN)
    - Identify top accident-related risk factors
    - Present interactive visual insights & model comparison
    
    **Dataset:** Crash Reporting - Drivers Data  
    Source: Kaggle / Public Portal
    
    **Author:** Narmatha A/P Kathiravan  
    **Course:** CPT6314 Final Year Project
    """)

# ---------- Dataset Details ----------
elif page == "📊 Dataset Details":
    st.header("Dataset Overview")
    st.write(df.head())
    st.subheader("Column Types and Missing Values")
    st.write(df.isnull().sum())
    st.write(df.dtypes)

    st.subheader("Summary Statistics")
    st.write(df.describe(include='all'))

# ---------- Data Insights ----------
elif page == "📈 Data Insights":
    st.header("Exploratory Data Analysis")
    st.subheader("Distribution of Injury Severity")
    sns.countplot(data=df, x='Injury Severity')
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("Correlation Heatmap (Accident-Related)")
    corr_cols = accident_features + [target_col]
    sns.heatmap(df_clean[corr_cols].corr(), annot=False, cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("Model Performance Comparison")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs'),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "ANN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=42)
    }

    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        res = evaluate_model(model, X_test, y_test)
        model_scores[name] = [res['accuracy'], res['precision'], res['recall'], res['f1_score']]

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10,6))
    for i, (name, scores) in enumerate(model_scores.items()):
        ax.bar(x + width*i - width*1.5, scores, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance Comparison')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Feature Importance (Random Forest - Accident Features)")
    rf_model = models["Random Forest"]
    rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    rf_filtered = rf_importances[accident_features].sort_values(ascending=False)
    sns.barplot(x=rf_filtered.values, y=rf_filtered.index)
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("Detailed Classification Report (XGBoost)")
    report = evaluate_model(models["XGBoost"], X_test, y_test)['classification_report']
    st.dataframe(pd.DataFrame(report).transpose())





