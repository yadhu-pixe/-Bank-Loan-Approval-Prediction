
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

import dataframe_image as dfi

# =========================
# 1. Load Dataset
# =========================
data = pd.read_csv('loan_data.csv')

input_sample = data.head(5)
print("Input Data Samples:\n", input_sample)
input_sample.to_csv('input_data_sample.csv', index=False)
dfi.export(input_sample, "input_data_sample.png")

# =========================
# 2. Initial Exploration
# =========================
print("Columns:", data.columns)
print("Shape:", data.shape)
print("Sample Data:\n", data.sample(5))
print("Description:\n", data.describe())


# Log Loan Amount
data['loanAmount_log'] = np.log(data['LoanAmount'])
plt.figure(figsize=(8, 5))
data['loanAmount_log'].hist(bins=20)
plt.title('Distribution of Log Loan Amount')
plt.xlabel('Log Loan Amount')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('log_loan_amount_histogram.png')
plt.show()

# Log Total Income
data['total_income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['total_income_log'] = np.log(data['total_income'])
plt.figure(figsize=(8, 5))
data['total_income_log'].hist(bins=20)
plt.title('Distribution of Log Total Income')
plt.xlabel('Log Total Income')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('log_total_income_histogram.png')
plt.show()

# =========================
# 4. Handle Missing Values
# =========================
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mean())
data['loanAmount_log'] = data['loanAmount_log'].fillna(data['loanAmount_log'].mean())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])

# =========================
# 5. Prepare Features & Target
# =========================
X = data.drop(['ApplicantIncome', 'CoapplicantIncome', 'Loan_Status'], axis=1)
Y = data['Loan_Status']

# Label Encoding
encoder = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = encoder.fit_transform(X[col])
Y = encoder.fit_transform(Y)


preprocessed_sample = pd.DataFrame(X, columns=data.drop(['ApplicantIncome', 'CoapplicantIncome', 'Loan_Status'], axis=1).columns)
preprocessed_sample.head(5).to_csv('preprocessed_features_sample.csv', index=False)
dfi.export(preprocessed_sample.head(5), "preprocessed_features_sample.png")


scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# 6. Train-Test Split
# =========================
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# =========================
# 7. Train Models
# =========================

rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train.ravel())
rfc_pred = rfc.predict(X_test)

# SVM
svm_model = SVC()
svm_model.fit(X_train, Y_train.ravel())
svm_pred = svm_model.predict(X_test)

# =========================
# 8. Save Predictions
# =========================
predictions_df = pd.DataFrame({
    'Actual': Y_test,
    'RandomForest_Pred': rfc_pred,
    'SVM_Pred': svm_pred
})
predictions_df.head(10).to_csv('predictions_sample.csv', index=False)
dfi.export(predictions_df.head(10), "predictions_sample.png")

# =========================
# 9. Evaluate Models
# =========================
print("Random Forest Accuracy:", accuracy_score(Y_test, rfc_pred))
print("SVM Accuracy:", accuracy_score(Y_test, svm_pred))

cm = confusion_matrix(Y_test, svm_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Rejected', 'Approved'],
            yticklabels=['Rejected', 'Approved'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for SVM Model')
plt.savefig('svm_confusion_matrix.png')
plt.show()

# Confusion Matrix Random Forest
cm_rfc = confusion_matrix(Y_test, rfc_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rfc, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Rejected', 'Approved'],
            yticklabels=['Rejected', 'Approved'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Random Forest Model')
plt.savefig('rfc_confusion_matrix.png')
plt.show()
