CREDIT CARD FRAUD DETECTION SYSTEM
This project is a complete machine learning pipeline built to detect fraudulent credit card transactions using advanced models and real-time logic. It focuses on high accuracy and minimal false positives, making online transactions safer and smarter.

TABLE OF CONTENTS:

Features

Tech Stack

Usage

Model Training

Results


FEATURES:

Real-time fraud detection logic

Class imbalance handled using SMOTE

Multiple ML models implemented: Logistic Regression, Random Forest, XGBoost, Isolation Forest

Evaluation using F1 Score, ROC-AUC, and Confusion Matrix

Insightful visualizations to understand fraud patterns

Clean and modular code — easy to extend or integrate into other systems

TECH STACK:

Language: Python 3.x

Libraries: scikit-learn, pandas, matplotlib, seaborn, imbalanced-learn, xgboost

USAGE:
You can run the script directly to execute the complete pipeline - from loading the dataset and preprocessing to training, evaluating, and visualizing the models.
The output includes printed metrics for each model, visual plots for data exploration, and a sample prediction function that can be customized for real-time use cases.

MODEL TRAINING:

Dataset: Credit Card Fraud Dataset from Kaggle

PREPROCESSING:

Feature scaling

SMOTE for class balancing

MODELS USED:

Logistic Regression

Random Forest

XGBoost

Isolation Forest (unsupervised)

EVALUATION:

F1 Score

Confusion Matrix

ROC-AUC

RESULTS:

Logistic Regression: 97.2% accuracy, F1 score 0.83, ROC-AUC 0.91

Random Forest: 99.4% accuracy, F1 score 0.92, ROC-AUC 0.98

XGBoost: 99.5% accuracy, F1 score 0.93, ROC-AUC 0.99

Isolation Forest: 94.1% accuracy, F1 score 0.67, ROC-AUC 0.85

XGBoost provided the most reliable detection performance, especially for imbalanced fraud cases
