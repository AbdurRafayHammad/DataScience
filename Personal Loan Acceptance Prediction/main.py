

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# STEP 1: LOAD DATA

df = pd.read_csv("bank.csv")   

print("Dataset Loaded Successfully!\n")
print(df.head())

# STEP 2: BASIC INFO

print("\nColumns:\n", df.columns)
print("\nInfo:\n")
print(df.info())

# STEP 3: DATA VISUALIZATION

# Target distribution
sns.countplot(x='deposit', data=df)
plt.title("Loan Acceptance Distribution")
plt.show()

# Age distribution
sns.histplot(df['age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Job vs Deposit
plt.figure(figsize=(12,5))
sns.countplot(x='job', hue='deposit', data=df)
plt.xticks(rotation=45)
plt.title("Job vs Deposit")
plt.show()

# Marital vs Deposit
sns.countplot(x='marital', hue='deposit', data=df)
plt.title("Marital Status vs Deposit")
plt.show()

# STEP 4: DATA PREPROCESSING

df_encoded = df.copy()
le = LabelEncoder()

for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

print("\nData Encoding Done!")

# STEP 5: SPLIT DATA

X = df_encoded.drop('deposit', axis=1)
y = df_encoded['deposit']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nData Split Done!")

# STEP 6: MODEL TRAINING

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# STEP 7: PREDICTION

y_pred = model.predict(X_test)

# STEP 8: EVALUATION

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# STEP 9: FEATURE IMPORTANCE


importances = model.feature_importances_
feature_names = X.columns

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10,5))
feat_imp.head(10).plot(kind='bar')
plt.title("Top 10 Important Features")
plt.show()


