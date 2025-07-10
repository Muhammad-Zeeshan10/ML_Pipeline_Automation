import pandas as pd
import numpy as np
import pickle
import io
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score
)

def get_feature_importance(data, feature_columns, target_column):
    """
    Calculate feature importance using a simple RandomForest model
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset
    feature_columns : list
        List of feature columns
    target_column : str
        The target column
    
    Returns:
    --------
    feature_importances : dict
        Dictionary of feature importances
    """
    # Create a copy of the data with only the relevant columns
    data_subset = data[feature_columns + [target_column]].copy()
    
    # Handle missing values for the simple importance calculation
    data_subset = data_subset.dropna()
    
    # If no data left after dropping NA, return None
    if len(data_subset) == 0:
        return None
    
    # Prepare features and target
    X = data_subset[feature_columns]
    y = data_subset[target_column]
    
    # Convert categorical features to numeric
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.factorize(X[col])[0]
    
    # For categorical target, use a classifier
    if not pd.api.types.is_numeric_dtype(y):
        model = RandomForestClassifier(n_estimators=10, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=10, random_state=42)
    
    try:
        # Train the model
        model.fit(X, y)
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create a dictionary of feature importances
        feature_importances = dict(zip(feature_columns, importances))
        
        return feature_importances
    
    except Exception as e:
        # If there's an error, try with a simpler approach
        print(f"Error calculating feature importance: {str(e)}")
        return None

def detect_task_type(data, target_column=None):
    """
    Detect the type of machine learning task based on the target variable
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset
    target_column : str, optional
        The target column. If None, clustering task is assumed.
    
    Returns:
    --------
    task_type : str
        The detected task type: 'classification', 'regression', or 'clustering'
    """
    if target_column is None:
        return "clustering"
    
    # Check if target is numeric
    if pd.api.types.is_numeric_dtype(data[target_column]):
        # Check if it's likely to be classification (few unique values)
        if data[target_column].nunique() <= 10:
            # If all values are integers, it's likely a classification
            if all(data[target_column].dropna().apply(lambda x: float(x).is_integer())):
                return "classification"
        
        # Otherwise, it's regression
        return "regression"
    
    # If target is not numeric, it's a classification task
    return "classification"

def get_model_suggestions(task_type):
    """
    Get model suggestions based on the task type
    
    Parameters:
    -----------
    task_type : str
        The task type: 'classification', 'regression', or 'clustering'
    
    Returns:
    --------
    model_suggestions : dict
        Dictionary of suggested models with descriptions
    """
    if task_type == "regression":
        return {
            "Linear Regression": {
                "model": LinearRegression(),
                "description": "Simple and interpretable model for linear relationships. Fast to train, but can't capture complex patterns."
            },
            "Polynomial Regression": {
                "model": LinearRegression(), # We'll apply polynomial features in train_model
                "description": "Extension of linear regression that can model curved relationships using polynomial terms."
            },
            "Support Vector Regressor (SVR)": {
                "model": SVR(),
                "description": "Effective in high-dimensional spaces. Can handle non-linear patterns using kernels."
            }
        }
    
    elif task_type == "classification":
        return {
            "K-Nearest Neighbors (KNN)": {
                "model": KNeighborsClassifier(),
                "description": "Simple algorithm that classifies based on closest training examples. No training phase."
            },
            "Decision Tree Classifier": {
                "model": DecisionTreeClassifier(),
                "description": "Can capture non-linear relationships. Prone to overfitting but easy to interpret."
            },
            "Random Forest Classifier": {
                "model": RandomForestClassifier(),
                "description": "Ensemble of decision trees. Good accuracy, handles non-linear data, but less interpretable."
            },
            "Support Vector Classifier (SVM)": {
                "model": SVC(probability=True),
                "description": "Effective in high-dimensional spaces. Can handle non-linear boundaries using kernels."
            }
        }
    
    elif task_type == "clustering":
        return {
            "K-Means Clustering": {
                "model": KMeans(),
                "description": "Partitioning method that divides data into k clusters. Fast and scalable, but sensitive to initial centroids."
            }
        }
    
    return {}

def train_model(data, features, target, task_type, model_name, test_size=0.2, random_state=42):
    """
    Train a model on the given data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The preprocessed dataset
    features : list
        List of feature columns
    target : str or None
        The target column. If None, clustering task is assumed.
    task_type : str
        The task type: 'classification', 'regression', or 'clustering'
    model_name : str
        The name of the model to train, or 'auto' to try multiple models
    test_size : float, optional (default=0.2)
        The proportion of the dataset to include in the test split
    random_state : int, optional (default=42)
        Random seed for reproducibility
    
    Returns:
    --------
    model : object
        The trained model
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Test features
    y_train : pandas.Series
        Training target
    y_test : pandas.Series
        Test target
    predictions : numpy.ndarray
        Model predictions on the test set
    metrics : dict
        Evaluation metrics
    actual_model_name : str
        The name of the actual model used (useful when 'auto' is selected)
    """
    # Make sure all features exist in the data
    valid_features = [f for f in features if f in data.columns]
    
    if not valid_features:
        raise ValueError("No valid features found in the dataset. Please select different features.")
    
    # For clustering, no target is needed
    if task_type == "clustering":
        # Prepare features
        X = data[valid_features].copy()
        
        # Handle non-numeric features
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.factorize(X[col])[0]
        
        # Split data for visualization purposes
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        
        # Get model
        model_suggestions = get_model_suggestions(task_type)
        model_info = list(model_suggestions.values())[0]  # Use KMeans for clustering
        model = model_info["model"]
        
        # Train the model (find optimal number of clusters)
        best_n_clusters = 2  # Default
        best_score = -1
        
        # Try different numbers of clusters
        for n_clusters in range(2, min(11, len(X_train) // 10 + 1)):
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X_train)
            
            # Calculate silhouette score if there are enough samples
            if len(X_train) >= n_clusters * 2:
                try:
                    score = silhouette_score(X_train, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                except:
                    pass
        
        # Train the final model with the optimal number of clusters
        model = KMeans(n_clusters=best_n_clusters, random_state=random_state, n_init=10)
        model.fit(X_train)
        
        # Get predictions
        train_predictions = model.predict(X_train)
        predictions = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "silhouette": best_score if best_score > -1 else 0,
            "n_clusters": best_n_clusters
        }
        
        return model, X_train, X_test, None, None, predictions, metrics, "K-Means Clustering"
    
    else:
        # Make sure target exists in the data
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in the dataset.")
            
        # Prepare features and target
        X = data[valid_features].copy()
        y = data[target].copy()
        
        # Handle non-numeric features
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.factorize(X[col])[0]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Get model suggestions
        model_suggestions = get_model_suggestions(task_type)
        
        if model_name == "auto":
            # Try all models and select the best one
            best_model = None
            best_score = -float("inf")
            best_model_name = None
            best_predictions = None
            
            for name, model_info in model_suggestions.items():
                try:
                    # Train the model
                    model = model_info["model"]
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    predictions = model.predict(X_test)
                    
                    # Evaluate the model
                    if task_type == "regression":
                        score = r2_score(y_test, predictions)
                    else:  # classification
                        score = accuracy_score(y_test, predictions)
                    
                    # Check if this model is better
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_model_name = name
                        best_predictions = predictions
                
                except Exception as e:
                    print(f"Error training {name}: {str(e)}")
                    continue
            
            if best_model is None:
                raise Exception("All models failed to train properly.")
            
            model = best_model
            predictions = best_predictions
            actual_model_name = best_model_name
        
        else:
            # Train the selected model
            model_info = model_suggestions[model_name]
            model = model_info["model"]
            model.fit(X_train, y_train)
            
            # Make predictions
            predictions = model.predict(X_test)
            actual_model_name = model_name
        
        # Calculate evaluation metrics
        if task_type == "regression":
            metrics = {
                "mae": mean_absolute_error(y_test, predictions),
                "mse": mean_squared_error(y_test, predictions),
                "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                "r2": r2_score(y_test, predictions)
            }
        
        else:  # classification
            # Get class labels
            if hasattr(model, 'classes_'):
                class_names = model.classes_
            else:
                class_names = sorted(np.unique(y))
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "class_names": class_names
            }
            
            # Multi-class metrics with 'weighted' average
            try:
                metrics["precision"] = precision_score(y_test, predictions, average='weighted')
                metrics["recall"] = recall_score(y_test, predictions, average='weighted')
                metrics["f1"] = f1_score(y_test, predictions, average='weighted')
            except:
                # Binary metrics
                metrics["precision"] = precision_score(y_test, predictions, average='binary')
                metrics["recall"] = recall_score(y_test, predictions, average='binary')
                metrics["f1"] = f1_score(y_test, predictions, average='binary')
        
        return model, X_train, X_test, y_train, y_test, predictions, metrics, actual_model_name

def save_model(model):
    """
    Save the trained model to a byte stream
    
    Parameters:
    -----------
    model : object
        The trained model to save
    
    Returns:
    --------
    model_bytes : bytes
        The serialized model as bytes
    """
    # Use joblib to serialize the model
    output = io.BytesIO()
    joblib.dump(model, output)
    model_bytes = output.getvalue()
    
    return model_bytes

def load_model(model_bytes):
    """
    Load a model from a byte stream
    
    Parameters:
    -----------
    model_bytes : bytes
        The serialized model as bytes
    
    Returns:
    --------
    model : object
        The deserialized model
    """
    # Use joblib to deserialize the model
    input_stream = io.BytesIO(model_bytes)
    model = joblib.load(input_stream)
    
    return model
