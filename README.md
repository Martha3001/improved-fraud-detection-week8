# Fraud Detection System

## Project Overview

This repository contains a comprehensive fraud detection system  designed to identify fraudulent transactions in both e-commerce and credit card datasets. The system leverages machine learning techniques to detect suspicious activity while minimizing false positives.

### Key Features
- Data preprocessing pipelines for handling imbalanced datasets
- Feature engineering for transaction pattern recognition
- Machine learning models (Logistic Regression and XGBoost) with performance comparison
- Model explainability using SHAP values
- Comprehensive evaluation metrics tailored for fraud detection

## Project Structure

```
fraud-detection-system/
│
├── data/
│   ├── raw/                   # Original dataset files
│   │   ├── Fraud_Data.csv
│   │   ├── IpAddress_to_Country.csv
│   │   └── creditcard.csv
│   └── processed/             # Processed and cleaned data
│
├── notebooks/
│   ├── data_analysis_and_processing.ipynb
│   └── models.ipynb
│
├── src/
│   ├── data_analysis.py       # Data cleaning and EDA functions
│   └── models.py             # Model training and evaluation functions
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup Instructions

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Martha3001/improved-fraud-detection-week8.git
   cd improved-fraud-detection-week8
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Analysis

1. **Data Processing and EDA**:
   ```bash
   jupyter notebook notebooks/data_analysis_and_processing.ipynb
   ```
   - This notebook performs data cleaning, feature engineering, and exploratory analysis

2. **Model Training and Evaluation**:
   ```bash
   jupyter notebook notebooks/models.ipynb
   ```
   - This notebook covers model training, evaluation, and SHAP analysis

## Data Description

### E-commerce Transaction Data (Fraud_Data.csv)
- **user_id**: Unique user identifier
- **signup_time**: Timestamp of user registration
- **purchase_time**: Transaction timestamp
- **purchase_value**: Transaction amount in dollars
- **device_id**: Device identifier
- **source**: Traffic source (SEO, Ads, Direct)
- **browser**: Browser used for transaction
- **sex**: User gender (M/F)
- **age**: User age
- **ip_address**: IP address of transaction
- **class**: Target variable (1 = fraud, 0 = legitimate)

### Credit Card Data (creditcard.csv)
- **Time**: Seconds elapsed since first transaction
- **V1-V28**: Anonymized PCA-transformed features
- **Amount**: Transaction amount
- **Class**: Target variable (1 = fraud, 0 = legitimate)

## Key Results

### Model Performance

**E-commerce Fraud Detection:**
| Model               | Precision | Recall | F1 Score | AUC Score |
|---------------------|-----------|--------|----------|-----------|
| Logistic Regression | 0.10      | 0.50   | 0.16     | 0.51      |
| XGBoost             | 0.96      | 0.54   | 0.69     | 0.84      |

**Credit Card Fraud Detection:**
| Model               | Precision | Recall | F1 Score | AUC Score |
|---------------------|-----------|--------|----------|-----------|
| Logistic Regression | 0.75      | 0.70   | 0.72     | 0.92      |
| XGBoost             | 0.94      | 0.76   | 0.84     | 0.93      |
