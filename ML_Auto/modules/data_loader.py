import pandas as pd
import numpy as np
import streamlit as st
import io

def load_data(file):
    
    file_name = file.name.lower()
    
    try:
        if file_name.endswith('.csv'):
            # Try different encodings and delimiters for CSV
            try:
                data = pd.read_csv(file)
            except UnicodeDecodeError:
                data = pd.read_csv(file, encoding='latin1')
            except:
                # Try with different delimiter
                data = pd.read_csv(file, sep=';')
                
            return data, 'CSV'
            
        elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            data = pd.read_excel(file)
            return data, 'Excel'
            
        elif file_name.endswith('.json'):
            data = pd.read_json(file)
            return data, 'JSON'
            
        else:
            raise ValueError("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
    
    except Exception as e:
        raise Exception(f"Error loading file: {str(e)}")

def get_data_preview(data, rows=5):
    
    return data.head(rows)

def get_data_summary(data):
    
    summary = {
        'shape': data.shape,
        'missing_values': data.isnull().sum().sum(),
        'dtypes': data.dtypes,
        'numeric_columns': data.select_dtypes(include=['number']).columns.tolist(),
        'categorical_columns': data.select_dtypes(exclude=['number']).columns.tolist()
    }
    
    return summary
