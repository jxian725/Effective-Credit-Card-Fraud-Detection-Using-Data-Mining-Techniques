# Credit Card Fraud Detection with Data Mining Techniques

This repository contains all the Jupyter Notebook files and scripts for my research project on **credit card fraud detection**, conducted using the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/) dataset. The goal was to evaluate and compare multiple machine learning models and assess the impact of feature engineering through cumulative importance-based feature selection.

## 🔍 Objective

Identify the most effective machine learning algorithm for detecting fraudulent credit card transactions, and explore whether reducing the dataset using feature importance can improve performance or efficiency without compromising accuracy.

## 📂 Project Structure

- `datasets/` – Datasets used for all 5 models taken from Kaggle.
- `models/` – Saved model files for all 5 models for each fold.
- `feature_importance/` – CSV files containing gain-based feature importance from LightGBM.
- `10_fold_indices.pkl` – Pre-generated 10-fold cross-validation indices to ensure consistent evaluation across models.

## 🧠 Models Implemented

| Model               | Encoding           | Preprocessing        | Notes                                    |
|--------------------|--------------------|----------------------|------------------------------------------|
| Logistic Regression| Target Encoding    | Standard Scaling     | Early stopping (patience=50, max_iter=5000) |
| Random Forest      | One-Hot Encoding   | Standard Scaling     | Trained with default hyperparameters     |
| XGBoost            | Label Encoding     | Standard Scaling     | GPU-based training with early stopping   |
| LightGBM           | Target Encoding    | Standard Scaling     | Selected as best model, used for feature engineering |
| Deep Neural Network| Embedding Encoding | Standard Scaling     | 4 hidden layers, trained with early stopping |

## 📊 Evaluation Metrics

For each model, the following evaluation metrics were computed across 10 folds:

- AUC-ROC
- Accuracy
- Precision (class 0 and 1)
- Recall (class 0 and 1)
- F1 Score (class 0 and 1)

## 🧪 Feature Engineering

LightGBM’s gain-based feature importance was used to compute cumulative importance thresholds:
- **95% Threshold** retained 162 features
- **86% Threshold** retained 80 features

Retrained models using these subsets showed the 95% threshold consistently matched or outperformed the full dataset in key metrics, especially in precision and F1-score for fraudulent transactions.

## 📁 Dataset

The dataset used in this project is the [IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/), which contains anonymized transactional and identity features.

## ✅ Requirements

- Python 3.10.15
- LightGBM
- XGBoost
- scikit-learn
- category_encoders
- PyTorch
- pandas, numpy, matplotlib

