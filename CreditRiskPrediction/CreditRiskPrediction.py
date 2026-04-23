import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------
# STEP 1: Load dataset
# -------------------------------
df = pd.read_csv("loan_approval_dataset.csv")

# FIX column names (IMPORTANT)
df.columns = df.columns.str.strip()

print("Columns:", df.columns)

# -------------------------------
# STEP 2: Check missing values
# -------------------------------
print("\nMissing values:")
print(df.isnull().sum())

# -------------------------------
# STEP 3: Visualization (EDA)
# -------------------------------

# Loan amount distribution
sns.histplot(df['loan_amount'], kde=True)
plt.title("Loan Amount Distribution")
plt.show()

# Income vs Loan
sns.scatterplot(x='income_annum', y='loan_amount', data=df)
plt.title("Income vs Loan")
plt.show()

# Education vs Loan Status
sns.countplot(x='education', hue='loan_status', data=df)
plt.title("Education vs Loan Status")
plt.show()

# -------------------------------
# STEP 4: Convert categorical → numeric
# -------------------------------
df['education'] = df['education'].map({' Graduate':1, ' Not Graduate':0})
df['self_employed'] = df['self_employed'].map({' Yes':1, ' No':0})
df['loan_status'] = df['loan_status'].map({' Approved':1, ' Rejected':0})

# -------------------------------
# STEP 5: Features & Target
# -------------------------------
X = df.drop(['loan_id', 'loan_status'], axis=1)
y = df['loan_status']

# -------------------------------
# STEP 6: Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------------------------------
# STEP 7: Train model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# STEP 8: Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# STEP 9: Evaluation
# -------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))