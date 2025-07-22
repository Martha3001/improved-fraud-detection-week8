# Data Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import os

def load_data(file_path):
    """
    Load a CSV file into a pandas DataFrame.
    
    Parameters:
        file_path (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    # if not os.path.exists(file_path):
    #     raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    return pd.read_csv(file_path)

def check_data_quality(df):
    """
    Check the quality of the data in a pandas DataFrame.
    
    This function provides an overview of the DataFrame, including its shape,
    descriptive statistics, data types, missing values, and duplicate rows.
    Parameters:
        - df (pd.DataFrame): The DataFrame to analyze.
    Raises:
        - ValueError: If the input is not a pandas DataFrame.
    Returns:
        - None
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    print("=== Data Overview ===")
    print(f"Shape: {df.shape}")
    print("=== Data Description ===")
    print(df.describe())
    print("=== Data Types ===")
    print(df.dtypes)
    print("\n=== Missing Values ===")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values.")

    print("\n=== Duplicate Rows ===")
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"{duplicate_count} duplicate row(s) found.")
    else:
        print("No duplicate rows found.")


def convert_to_datetime(df, columns):
    """
    Converts specified columns in a DataFrame to datetime format.

    Parameters:
    - df (pd.DataFrame): The DataFrame to modify.
    - columns (list): List of column names to convert.

    Returns:
    - pd.DataFrame: A new DataFrame with specified columns converted to datetime.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
    return df


# plot pie chart for column
def plot_pie_chart(df, column, title=None):
    """
    Plots a pie chart for a specified column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column (str): The column to plot.
    - title (str, optional): Title of the pie chart.

    Returns:
    - None
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    plt.figure(figsize=(8, 6))
    df[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(title if title else f"Pie Chart of {column}")
    plt.ylabel('')
    plt.show()


# Plotting bar chart for column sorted top 20 values horizontally
def plot_bar_chart(df, column, title=None):
    """
    Plots a horizontal bar chart for a specified column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column (str): The column to plot.
    - title (str, optional): Title of the bar chart.

    Returns:
    - None
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    plt.figure(figsize=(10, 6))
    df[column].value_counts().nlargest(20).sort_values().plot.barh()
    plt.title(title if title else f"Bar Chart of {column}")
    plt.xlabel('Count')
    plt.ylabel(column)
    plt.show()


## Plotting Histograms
def plot_histogram(df, column, title=None):
    """
    Plots a histogram for a specified column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column (str): The column to plot.
    - title (str, optional): Title of the histogram.

    Returns:
    - None
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(title if title else f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


def analyze_class_relationships(df):
    """
    Analyzes the relationship between the 'class' column and other specified variables.

    Args:
        df (pd.DataFrame): The input DataFrame with 'class' and other relevant columns.
    """
    analysis_columns = ['purchase_value', 'age', 'source', 'browser', 'sex']
    numerical_cols = ['purchase_value', 'age']
    categorical_cols = ['source', 'browser', 'sex']

    print("=== Analyzing Class Relationships ===")

    for col in analysis_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
            continue

        print(f"\n--- Analysis for {col} ---")

        if col in numerical_cols:
            # Numerical columns: box plot

            plt.figure(figsize=(8, 6))
            sns.boxplot(x='class', y=col, data=df)
            plt.title(f'{col} Distribution by Class')
            plt.xlabel('Class')
            plt.ylabel(col)
            plt.show()

        elif col in categorical_cols:
            # Categorical columns:  count plot

            plt.figure(figsize=(10, 6))
            sns.countplot(x=col, hue='class', data=df)
            plt.title(f'Count of {col} by Class')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.show()

