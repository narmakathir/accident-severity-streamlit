# --- Import Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# --- Streamlit Title ---
st.title("Accident Severity Prediction App")
st.write("Using ML Models: Random Forest, XGBoost, Logistic Regression, ANN")

# --- Download & Load Dataset from Google Drive ---
file_id = '1sVplp_5lFb3AMG5vWRqIltwNazLyM8vH'
output = 'data.csv'
url = f'https://drive.google.com/uc?id={file_id}'
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

df = pd.read_csv(output)
target_col = 'Injury Severity'

# --- Drop Irrelevant Columns ---
drop_cols = [
    'Report Number', 'Local Case Number', 'Person ID', 'Vehicle ID',
    'Latitude', 'Longitude', 'Location', 'Driverless Vehicle', 'Parked Vehicle'
]
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# --- Handle Missing Values ---
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# --- Encode Categorical Columns ---
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# --- Normalize Numeric Columns ---
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include='number').columns.difference([target_col])
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# --- Split Data ---
X = df.drop(target_col, axis=1)
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Helper Function for Evaluation ---
model_scores = {}
def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average='weighted') * 100
    rec = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100
    model_scores[name] = [acc, prec, rec, f1]

# --- Train and Evaluate Models ---
with st.spinner("Training models..."):
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    evaluate_model(rf_model, "Random Forest")

    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    evaluate_model(xgb_model, "XGBoost")

    log_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    log_model.fit(X_train, y_train)
    evaluate_model(log_model, "Logistic Regression")

    ann_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=42)
    ann_model.fit(X_train, y_train)
    evaluate_model(ann_model, "Artificial Neural Network")

# --- Display Scores ---
st.subheader("Model Comparison")
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
scores = pd.DataFrame(model_scores, index=metrics).T
st.dataframe(scores.style.format("{:.2f}"))

# --- Visualize Comparison ---
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.2
ax.bar(x - 1.5*width, model_scores['Random Forest'], width, label='RF')
ax.bar(x - 0.5*width, model_scores['XGBoost'], width, label='XGB')
ax.bar(x + 0.5*width, model_scores['Logistic Regression'], width, label='LR')
ax.bar(x + 1.5*width, model_scores['Artificial Neural Network'], width, label='ANN')
ax.set_ylabel('Score (%)')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
st.pyplot(fig)

# --- Correlation Heatmap ---
accident_features = [
    'Driver At Fault', 'Circumstance', 'Driver Distracted By', 'Collision Type',
    'Vehicle Movement', 'Vehicle Going Dir', 'Vehicle First Impact Location',
    'Vehicle Damage Extent', 'Vehicle Body Type', 'Traffic Control',
    'Weather', 'Surface Condition', 'Light', 'Speed Limit', 'Driver Substance Abuse'
]
features_in_df = [col for col in accident_features if col in df.columns]
if features_in_df:
    st.subheader("Correlation Heatmap (Accident-Related Features)")
    corr_matrix = df[features_in_df + [target_col]].corr()
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
    st.pyplot(fig2)

