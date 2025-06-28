import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('your_data.csv')  # Replace with your data source

# Check class distribution
print("Original Class Distribution:")
print(df['Injury Severity'].value_counts())

# Visualize distribution
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='Injury Severity', order=['minor', 'serious', 'fatal'], palette='coolwarm')
plt.title('Original Injury Severity Distribution')
plt.show()

# Preprocessing
# Separate features and target
X = df.drop('Injury Severity', axis=1)
y = df['Injury Severity'].map({'minor':0, 'serious':1, 'fatal':2})  # Encoding

# Identify categorical and numerical features
cat_features = X.select_dtypes(include=['object']).columns
num_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

# Handle class imbalance (especially focusing on 'minor' class)
smote = SMOTE(sampling_strategy={0: X.shape[0]//3,  # minor
                                1: X.shape[0]//3,  # serious
                                2: X.shape[0]//3}, # fatal
              random_state=42)

# Create model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('sampler', smote),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',  # Extra weight for minority classes
        random_state=42
    ))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['minor', 'serious', 'fatal']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['minor', 'serious', 'fatal'],
            yticklabels=['minor', 'serious', 'fatal'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature importance (for numerical features)
if len(num_features) > 0:
    importances = model.named_steps['classifier'].feature_importances_[:len(num_features)]
    feat_importances = pd.Series(importances, index=num_features)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Top Numerical Feature Importances')
    plt.show()
