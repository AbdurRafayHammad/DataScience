# =========================================================
# TASK 1: TERM DEPOSIT SUBSCRIPTION PREDICTION
# =========================================================

# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    roc_curve,
    auc
)

import shap

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("bank-additional-full.csv", sep=';')

# =========================
# EXPLORE DATASET
# =========================
print("First 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nTarget Variable Count")
print(df['y'].value_counts())

# =========================
# ENCODE CATEGORICAL FEATURES
# =========================
label_encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_encoder.fit_transform(df[col])

print("\nEncoded Dataset")
print(df.head())

# =========================
# SPLIT FEATURES AND TARGET
# =========================
X = df.drop('y', axis=1)
y = df['y']

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# LOGISTIC REGRESSION MODEL
# =========================================================
print("\n================ LOGISTIC REGRESSION ================\n")

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

# =========================
# EVALUATION
# =========================
print("Accuracy:",
      accuracy_score(y_test, lr_pred))

print("F1 Score:",
      f1_score(y_test, lr_pred))

print("\nClassification Report")
print(classification_report(y_test, lr_pred))

# =========================
# CONFUSION MATRIX
# =========================
cm_lr = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(6,5))

sns.heatmap(cm_lr,
            annot=True,
            fmt='d')

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =========================
# ROC CURVE
# =========================
lr_prob = lr_model.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(y_test, lr_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr,
         label='ROC Curve (AUC = %0.2f)' % roc_auc)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")

plt.legend()

plt.show()

# =========================================================
# RANDOM FOREST MODEL
# =========================================================
print("\n================ RANDOM FOREST ================\n")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

# =========================
# EVALUATION
# =========================
print("Accuracy:",
      accuracy_score(y_test, rf_pred))

print("F1 Score:",
      f1_score(y_test, rf_pred))

print("\nClassification Report")
print(classification_report(y_test, rf_pred))

# =========================
# CONFUSION MATRIX
# =========================
cm_rf = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,5))

sns.heatmap(cm_rf,
            annot=True,
            fmt='d')

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =========================
# ROC CURVE
# =========================
rf_prob = rf_model.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(y_test, rf_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr,
         label='ROC Curve (AUC = %0.2f)' % roc_auc)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest ROC Curve")

plt.legend()

plt.show()

# =========================================================
# SHAP EXPLAINABILITY
# =========================================================
print("\n================ SHAP EXPLANATIONS ================\n")

# Create SHAP explainer
explainer = shap.TreeExplainer(rf_model)

# Generate SHAP values
shap_values = explainer.shap_values(X_test)

# =========================
# GLOBAL FEATURE IMPORTANCE
# =========================
shap.summary_plot(
    shap_values,
    X_test
)

# =========================
# EXPLAIN 5 PREDICTIONS
# =========================
for i in range(5):

    print(f"\nExplaining Prediction {i+1}")

    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][i],
        X_test.iloc[i],
        matplotlib=True
    )

# =========================================================
# FEATURE IMPORTANCE
# =========================================================
importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop Features")
print(feature_importance.head(10))