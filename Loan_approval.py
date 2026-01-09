
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


# ==========================================================
# 1. LOAD DATASET
# ==========================================================
def load_dataset(file_path):
    data = pd.read_csv(file_path)

    # Save sample input table
    sample_data = data.head(5)
    sample_data.to_csv("input_data_sample.csv", index=False)
    dfi.export(sample_data, "input_data_sample.png")

    print("\nDataset Loaded Successfully")
    print("Shape:", data.shape)
    print("Columns:", list(data.columns))

    return data


# ==========================================================
# 2. FEATURE ENGINEERING & VISUALIZATION
# ==========================================================
def feature_engineering(data):

    # Log transform loan amount
    data["loanAmount_log"] = np.log(data["LoanAmount"])

    plt.figure(figsize=(8, 5))
    data["loanAmount_log"].hist(bins=20)
    plt.title("Distribution of Log Loan Amount")
    plt.grid(True)
    plt.savefig("log_loan_amount_histogram.png")
    plt.close()

    # Total income + log transform
    data["total_income"] = data["ApplicantIncome"] + data["CoapplicantIncome"]
    data["total_income_log"] = np.log(data["total_income"])

    plt.figure(figsize=(8, 5))
    data["total_income_log"].hist(bins=20)
    plt.title("Distribution of Log Total Income")
    plt.grid(True)
    plt.savefig("log_total_income_histogram.png")
    plt.close()

    return data


# ==========================================================
# 3. HANDLE MISSING VALUES
# ==========================================================
def handle_missing_values(data):

    categorical_cols = ["Gender", "Married", "Self_Employed",
                        "Dependents", "Loan_Amount_Term", "Credit_History"]

    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])

    data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].mean())
    data["loanAmount_log"] = data["loanAmount_log"].fillna(data["loanAmount_log"].mean())

    return data


# ==========================================================
# 4. PREPARE FEATURES & TARGET
# ==========================================================
def preprocess_features(data):

    # Drop unused columns
    X = data.drop(["ApplicantIncome", "CoapplicantIncome", "Loan_Status"], axis=1)
    Y = data["Loan_Status"]

    # Label Encoding
    encoder = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = encoder.fit_transform(X[col])

    Y = encoder.fit_transform(Y)

    # Save preprocessed feature sample
    dfi.export(X.head(5), "preprocessed_features_sample.png")
    X.head(5).to_csv("preprocessed_features_sample.csv", index=False)

    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, Y


# ==========================================================
# 5. TRAIN MODELS
# ==========================================================
def train_models(X_train, Y_train):

    # Random Forest
    rfc = RandomForestClassifier()
    rfc.fit(X_train, Y_train)

    # SVM
    svm = SVC()
    svm.fit(X_train, Y_train)

    return rfc, svm


# ==========================================================
# 6. EVALUATE MODELS
# ==========================================================
def evaluate_models(rfc, svm, X_test, Y_test):

    rfc_pred = rfc.predict(X_test)
    svm_pred = svm.predict(X_test)

    # Save output prediction sample
    predictions_df = pd.DataFrame({
        "Actual": Y_test,
        "RandomForest_Pred": rfc_pred,
        "SVM_Pred": svm_pred
    })

    dfi.export(predictions_df.head(10), "predictions_sample.png")
    predictions_df.head(10).to_csv("predictions_sample.csv", index=False)

    # Print accuracy values
    print("\nRandom Forest Accuracy:", accuracy_score(Y_test, rfc_pred))
    print("SVM Accuracy:", accuracy_score(Y_test, svm_pred))

    # Confusion Matrix (SVM)
    cm_svm = confusion_matrix(Y_test, svm_pred)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - SVM")
    plt.savefig("svm_confusion_matrix.png")
    plt.close()

    # Confusion Matrix (Random Forest)
    cm_rfc = confusion_matrix(Y_test, rfc_pred)
    sns.heatmap(cm_rfc, annot=True, fmt='d', cmap='Greens')
    plt.title("Confusion Matrix - Random Forest")
    plt.savefig("rfc_confusion_matrix.png")
    plt.close()


# ==========================================================
# MAIN EXECUTION PIPELINE
# ==========================================================
def main():

    # Step 1: Load Data
    data = load_dataset("loan_data.csv")

    # Step 2: Feature Engineering
    data = feature_engineering(data)

    # Step 3: Handle Missing Values
    data = handle_missing_values(data)

    # Step 4: Prepare Features & Target
    X, Y = preprocess_features(data)

    # Step 5: Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0)

    # Step 6: Train Models
    rfc, svm = train_models(X_train, Y_train)

    # Step 7: Evaluate & Generate Outputs
    evaluate_models(rfc, svm, X_test, Y_test)


if __name__ == "__main__":
    main()
