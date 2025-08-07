import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, f1_score, roc_auc_score,
                             confusion_matrix, accuracy_score, precision_score,
                             recall_score)
from imblearn.combine import SMOTETomek
import xgboost as xgb


def load_data(file_path):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path)


def split_data(df, target_column, test_size=0.3, random_state=42):
    """
    Splits the DataFrame into training and testing sets.

    Parameters:
    - df (pd.DataFrame): The DataFrame to split.
    - target_column (str): The name of the target column.
    - test_size (float): Proportion of the dataset to include
                        in the test split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X_train, X_test, y_train, y_test: Split data.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def handle_imbalanced_data(X_train, y_train):
    """
    Applies SMOTE-Tomek to the training data to handle class imbalance.

    Parameters:
    - X_train (pd.DataFrame): The training features.
    - y_train (pd.Series): The training labels.

    Returns:
    - X_resampled, y_resampled: The resampled training data.
    """
    smote_tomek = SMOTETomek(random_state=42)
    return smote_tomek.fit_resample(X_train, y_train)


def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model.

    Parameters:
    - X_train (pd.DataFrame): The training features.
    - y_train (pd.Series): The training labels.

    Returns:
    - LogisticRegression: The trained Logistic Regression model.
    """
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """
    Trains an XGBoost model.

    Parameters:
    - X_train (pd.DataFrame): The training features.
    - y_train (pd.Series): The training labels.

    Returns:
    - xgb.XGBClassifier: The trained XGBoost model.
    """
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, results_df,
                   dataset_name=None, model_name=None):
    """
    Evaluates the model using classification report and ROC AUC score.

    Parameters:
    - model: The trained model.
    - X_test (pd.DataFrame): The test features.
    - y_test (pd.Series): The test labels.

    Returns:
    - None
    """
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print("=== Classification Report ===")
    print(classification_report(y_test, preds))
    print("=== ROC AUC Score ===")
    print("ROC AUC Score:", roc_auc_score(y_test, probs))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, preds))

    if dataset_name and model_name:
        results_df = pd.concat([results_df, pd.DataFrame({
            'Dataset': [dataset_name],
            'Model': [model_name],
            'Accuracy': [round(accuracy_score(y_test, preds), 2)],
            'Precision': [round(precision_score(y_test, preds), 2)],
            'Recall': [round(recall_score(y_test, preds), 2)],
            'F1 Score': [round(f1_score(y_test, preds), 2)],
            'AUC Score': [round(roc_auc_score(y_test, probs), 2)]
        })], ignore_index=True)

    return results_df
