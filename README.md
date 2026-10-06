# customer-churn-prediction

This project is a machine learning pipeline designed to predict customer churn for a telecommunications company. It compares multiple machine learning algorithms to identify customers who are likely to leave the service.

## Project Overview
Goal: Predict customer churn (Yes/No) using historical customer data.
Algorithms Used: Logistic Regression, Decision Tree, and Random Forest.
Libraries Used: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Joblib.

## Dataset Features
The dataset contains customer demographic information, account details, and services signed up for:
Demographics: Gender, Senior Citizen, Partner, Dependents.
Account Info: Tenure, Contract type, Paperless Billing, Payment Method, Monthly Charges, Total Charges.
Services: Phone Service, Multiple Lines, Internet Service, Online Security, Tech Support, Streaming services.

# Workflow & Steps
Data Loading & Cleaning: Handled missing values (e.g., in `TotalCharges`) and removed duplicates and unnecessary columns (`customerID`).
Exploratory Data Analysis (EDA): Visualized churn distribution, tenure impact, monthly charges, and contract types using Seaborn and Matplotlib.
Data Preprocessing: Performed label encoding for binary columns, one-hot encoding for categorical features, and feature scaling using "StandardScaler".
Model Training & Evaluation: Trained and evaluated **Logistic Regression**, **Decision Tree**, and **Random Forest** classifiers.
                             Compared models using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
Model Deployment Preparation: Saved the trained Logistic Regression model, scaler, and feature list using `joblib` for future inference applications.

## Model Performance Comparison
Logistic Regression: High interpretability and stable performance on scaled features.
Decision Tree: Captures non-linear splits effectively with controlled max depth.
Random Forest: Ensemble method providing robust feature importance analysis and higher accuracy.

## How to Run the Code
1. Clone the repository.
2. Install the required libraries:
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
3.Run the Python script: python customer-churn-predictions.py
   
