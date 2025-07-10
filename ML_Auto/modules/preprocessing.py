import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

def detect_missing_values(data):
    
    missing_counts = data.isnull().sum()
    missing_percentage = (missing_counts / len(data) * 100).round(2)
    
    # Create a DataFrame with information about missing values
    missing_info = pd.DataFrame({
        'Feature': missing_counts.index,
        'Missing Values': missing_counts.values,
        'Missing Percentage': missing_percentage.values
    })
    
    # Filter to include only columns with missing values
    missing_info = missing_info[missing_info['Missing Values'] > 0].sort_values(
        by='Missing Values', ascending=False
    ).reset_index(drop=True)
    
    return missing_info

def handle_missing_values(data, column, method='Drop rows', constant_value=None):
    
    # Make a copy to avoid modifying the original data
    data_copy = data.copy()
    
    if method == 'Drop rows':
        # Drop rows with missing values in the specified column
        data_copy = data_copy.dropna(subset=[column])
    
    elif method == 'Fill with mean':
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(data_copy[column]):
            data_copy[column] = data_copy[column].fillna(data_copy[column].mean())
        else:
            # For non-numeric columns, use mode instead
            data_copy[column] = data_copy[column].fillna(data_copy[column].mode()[0])
    
    elif method == 'Fill with median':
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(data_copy[column]):
            data_copy[column] = data_copy[column].fillna(data_copy[column].median())
        else:
            # For non-numeric columns, use mode instead
            data_copy[column] = data_copy[column].fillna(data_copy[column].mode()[0])
    
    elif method == 'Fill with mode':
        # Use the most frequent value
        if not data_copy[column].mode().empty:
            data_copy[column] = data_copy[column].fillna(data_copy[column].mode()[0])
        else:
            # If mode is empty, use a default value
            data_copy[column] = data_copy[column].fillna('Unknown' if not pd.api.types.is_numeric_dtype(data_copy[column]) else 0)
    
    elif method == 'Fill with constant':
        # Fill with the specified constant value
        data_copy[column] = data_copy[column].fillna(constant_value)
    
    return data_copy

def encode_categorical_features(data, categorical_columns, method='One-Hot Encoding'):
    
    # Make a copy to avoid modifying the original data
    data_copy = data.copy()
    
    if method == 'One-Hot Encoding':
        # Apply one-hot encoding to each categorical column
        for col in categorical_columns:
            # Get one-hot encoded columns
            one_hot = pd.get_dummies(data_copy[col], prefix=col, drop_first=False)
            # Drop the original column
            data_copy = data_copy.drop(col, axis=1)
            # Join the encoded columns
            data_copy = data_copy.join(one_hot)
    
    elif method == 'Label Encoding':
        # Apply label encoding to each categorical column
        le = LabelEncoder()
        for col in categorical_columns:
            # Fill missing values if any
            data_copy[col] = data_copy[col].fillna('Unknown')
            # Convert to string (LabelEncoder requires string input)
            data_copy[col] = data_copy[col].astype(str)
            # Apply label encoding
            data_copy[col] = le.fit_transform(data_copy[col])
    
    return data_copy

def scale_features(data, numeric_columns, method='standard'):
    
    # Make a copy to avoid modifying the original data
    data_copy = data.copy()
    
    if method == 'standard':
        # Apply StandardScaler to numeric columns
        scaler = StandardScaler()
        data_copy[numeric_columns] = scaler.fit_transform(data_copy[numeric_columns])
    
    elif method == 'minmax':
        # Apply MinMaxScaler to numeric columns
        scaler = MinMaxScaler()
        data_copy[numeric_columns] = scaler.fit_transform(data_copy[numeric_columns])
    
    return data_copy

def detect_outliers(data, numeric_columns, method='IQR'):
    
    outliers_info = {}
    
    for col in numeric_columns:
        if method == 'IQR':
            # Calculate Q1 and Q3
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            # Calculate IQR
            IQR = Q3 - Q1
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Count outliers
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)].shape[0]
        
        elif method == 'Z-score':
            # Calculate Z-scores
            z_scores = (data[col] - data[col].mean()) / data[col].std()
            # Count outliers (|Z| > 3)
            outliers = data[abs(z_scores) > 3].shape[0]
        
        # Store outlier count
        outliers_info[col] = outliers
    
    return outliers_info

def remove_outliers(data, numeric_columns, method='IQR'):
    
    # Make a copy to avoid modifying the original data
    data_copy = data.copy()
    
    for col in numeric_columns:
        if method == 'IQR':
            # Calculate Q1 and Q3
            Q1 = data_copy[col].quantile(0.25)
            Q3 = data_copy[col].quantile(0.75)
            # Calculate IQR
            IQR = Q3 - Q1
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Remove outliers
            data_copy = data_copy[(data_copy[col] >= lower_bound) & (data_copy[col] <= upper_bound)]
        
        elif method == 'Z-score':
            # Calculate Z-scores
            z_scores = (data_copy[col] - data_copy[col].mean()) / data_copy[col].std()
            # Remove outliers (|Z| > 3)
            data_copy = data_copy[abs(z_scores) <= 3]
    
    return data_copy

