# ============================================================
# TELCO CUSTOMER CHURN PREDICTION 
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print("Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 2. DATA UNDERSTANDING
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nStatistical Summary:")
print(df.describe(include="all").T)


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())


# TotalCharges is sometimes stored as object
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nMissing Values After TotalCharges Conversion:")
print(df.isnull().sum())


# Remove missing rows
df.dropna(inplace=True)

print("\nShape After Removing Missing Values:")
print(df.shape)


# ============================================================
# 4. DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE CHECK")
print("=" * 60)

print("Duplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("Shape After Removing Duplicates:", df.shape)


# ============================================================
# 5. REMOVE CUSTOMER ID
# ============================================================

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)


# ============================================================
# 6. CHURN DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CHURN DISTRIBUTION")
print("=" * 60)

print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)


# ============================================================
# 7. EDA
# ============================================================

# Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Customer Count")
plt.show()


# Tenure vs Churn
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    kde=True
)
plt.title("Tenure vs Churn")
plt.show()


# Monthly Charges vs Churn
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)
plt.title("Monthly Charges vs Churn")
plt.show()


# Contract vs Churn
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)
plt.title("Contract Type vs Churn")
plt.xticks(rotation=20)
plt.show()


# Internet Service vs Churn
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)
plt.title("Internet Service vs Churn")
plt.show()


# Payment Method vs Churn
plt.figure(figsize=(10, 5))
sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn"
)
plt.title("Payment Method vs Churn")
plt.xticks(rotation=45)
plt.show()


# ============================================================
# 8. DATA PREPROCESSING
# ============================================================

# Gender Encoding
if "gender" in df.columns:
    df["gender"] = df["gender"].map({
        "Male": 1,
        "Female": 0
    })


# Yes/No Encoding
yes_no_columns = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn"
]

for col in yes_no_columns:
    if col in df.columns:
        df[col] = df[col].map({
            "Yes": 1,
            "No": 0
        })


# ============================================================
# 9. ONE-HOT ENCODING
# ============================================================

categorical_columns = df.select_dtypes(
    include="object"
).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)


# ============================================================
# 10. PROCESSED DATA
# ============================================================

print("\n" + "=" * 60)
print("PROCESSED DATA")
print("=" * 60)

print(df.head())

print("\nFinal Shape:")
print(df.shape)


# ============================================================
# 11. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(16, 10))

sns.heatmap(
    df.corr(),
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")
plt.show()


# ============================================================
# 12. X AND Y
# ============================================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# Save feature names
feature_names = X.columns.tolist()


# ============================================================
# 13. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ============================================================
# 14. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 15. LOGISTIC REGRESSION
# ============================================================

log_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

log_model.fit(
    X_train_scaled,
    y_train
)

y_pred_log = log_model.predict(X_test_scaled)

y_prob_log = log_model.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# 16. DECISION TREE
# ============================================================

dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

dt_model.fit(
    X_train,
    y_train
)

y_pred_dt = dt_model.predict(X_test)

y_prob_dt = dt_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 17. RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

y_pred_rf = rf_model.predict(X_test)

y_prob_rf = rf_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 18. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_test,
    y_pred,
    y_prob
):

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    return [
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]


# ============================================================
# 19. EVALUATE ALL MODELS
# ============================================================

log_results = evaluate_model(
    "LOGISTIC REGRESSION",
    y_test,
    y_pred_log,
    y_prob_log
)

dt_results = evaluate_model(
    "DECISION TREE",
    y_test,
    y_pred_dt,
    y_prob_dt
)

rf_results = evaluate_model(
    "RANDOM FOREST",
    y_test,
    y_pred_rf,
    y_prob_rf
)


# ============================================================
# 20. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        log_results[0],
        dt_results[0],
        rf_results[0]
    ],

    "Precision": [
        log_results[1],
        dt_results[1],
        rf_results[1]
    ],

    "Recall": [
        log_results[2],
        dt_results[2],
        rf_results[2]
    ],

    "F1 Score": [
        log_results[3],
        dt_results[3],
        rf_results[3]
    ],

    "ROC-AUC": [
        log_results[4],
        dt_results[4],
        rf_results[4]
    ]
})


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results)


# ============================================================
# 21. MODEL COMPARISON VISUALIZATION
# ============================================================

results_plot = results.set_index("Model")

results_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.legend()
plt.show()


# ============================================================
# 22. CONFUSION MATRIX - RANDOM FOREST
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred_rf
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")

plt.show()


# ============================================================
# 23. ROC CURVE
# ============================================================

fpr_log, tpr_log, _ = roc_curve(
    y_test,
    y_prob_log
)

fpr_dt, tpr_dt, _ = roc_curve(
    y_test,
    y_prob_dt
)

fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    y_prob_rf
)


plt.figure(figsize=(8, 6))

plt.plot(
    fpr_log,
    tpr_log,
    label="Logistic Regression"
)

plt.plot(
    fpr_dt,
    tpr_dt,
    label="Decision Tree"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label="Random Forest"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.show()


# ============================================================
# 24. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": rf_model.feature_importances_

})


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n" + "=" * 60)
print("TOP 15 IMPORTANT FEATURES")
print("=" * 60)

print(
    feature_importance.head(15)
)


plt.figure(figsize=(10, 7))

sns.barplot(
    data=feature_importance.head(15),
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 15 Important Features - Random Forest"
)

plt.show()


# ============================================================
# 25. SAVE LOGISTIC REGRESSION MODEL FOR STREAMLIT
# ============================================================

# We are intentionally deploying Logistic Regression
# because apps.py is designed for the scaled Logistic Regression model.
# ============================================================
# SAVE MODEL FILES IN CURRENT PROJECT FOLDER
# ============================================================

import os

project_folder = os.path.dirname(
    os.path.abspath(__file__)
)

model_path = os.path.join(
    project_folder,
    "telco_churn_logistic_regression.pkl"
)

scaler_path = os.path.join(
    project_folder,
    "telco_scaler.pkl"
)

features_path = os.path.join(
    project_folder,
    "telco_features.pkl"
)

joblib.dump(
    log_model,
    model_path
)

joblib.dump(
    scaler,
    scaler_path
)

joblib.dump(
    feature_names,
    features_path
)

print("\n" + "=" * 60)
print("MODEL FILES SAVED SUCCESSFULLY")
print("=" * 60)

print("Model:", model_path)
print("Scaler:", scaler_path)
print("Features:", features_path)


# ============================================================
# 26. TEST CUSTOMER
# ============================================================

def predict_churn(customer_data):

    # Create DataFrame
    new_customer = pd.DataFrame(
        [customer_data]
    )


    # Gender conversion
    if "gender" in new_customer.columns:

        new_customer["gender"] = new_customer[
            "gender"
        ].map({
            "Male": 1,
            "Female": 0
        })


    # Yes/No conversion
    yes_no_columns = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling"
    ]

    for col in yes_no_columns:

        if col in new_customer.columns:

            new_customer[col] = new_customer[
                col
            ].map({
                "Yes": 1,
                "No": 0
            })


    # One-hot encoding
    new_customer = pd.get_dummies(
        new_customer,
        drop_first=True,
        dtype=int
    )


    # Match training features
    new_customer = new_customer.reindex(
        columns=feature_names,
        fill_value=0
    )


    # Scaling
    new_customer_scaled = scaler.transform(
        new_customer
    )


    # Prediction
    prediction = log_model.predict(
        new_customer_scaled
    )[0]


    probability = log_model.predict_proba(
        new_customer_scaled
    )[0][1]


    # Result
    if prediction == 1:

        result = "YES - Customer may Churn"

    else:

        result = "NO - Customer is likely to Stay"


    print("\n" + "=" * 50)
    print("CUSTOMER CHURN PREDICTION")
    print("=" * 50)

    print("Prediction:", result)

    print(
        "Churn Probability:",
        round(probability * 100, 2),
        "%"
    )

    print(
        "Stay Probability:",
        round((1 - probability) * 100, 2),
        "%"
    )

    return prediction, probability


# ============================================================
# 27. TEST CUSTOMER DATA
# ============================================================

customer = {

    "gender": "Female",

    "SeniorCitizen": 0,

    "Partner": "No",

    "Dependents": "No",

    "tenure": 2,

    "PhoneService": "Yes",

    "MultipleLines": "No",

    "InternetService": "Fiber optic",

    "OnlineSecurity": "No",

    "OnlineBackup": "No",

    "DeviceProtection": "No",

    "TechSupport": "No",

    "StreamingTV": "Yes",

    "StreamingMovies": "Yes",

    "Contract": "Month-to-month",

    "PaperlessBilling": "Yes",

    "PaymentMethod": "Electronic check",

    "MonthlyCharges": 85.50,

    "TotalCharges": 171.00
}


# Run test prediction
predict_churn(customer)