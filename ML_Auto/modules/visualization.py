import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_correlation_matrix(data):
    """
    Plot correlation matrix heatmap
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset with numeric features
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    """
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create figure with appropriate size based on number of features
    n_features = len(corr_matrix.columns)
    
    # Make sure we have enough space for all features
    figsize = (max(12, n_features * 1.2), max(10, n_features * 1.0))
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate font size - smaller for more features
    fontsize = 10 if n_features <= 10 else max(4, 10 - (n_features-10)/4)
    
    # Create heatmap with all features visible
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        linewidths=0.5, 
        fmt=".2f",
        ax=ax,
        annot_kws={"size": fontsize}
    )
    
    # Rotate x-axis labels if many features
    if n_features > 10:
        plt.xticks(rotation=45, ha='right')
    
    plt.title('Correlation Matrix')
    plt.tight_layout()
    
    return fig

def plot_feature_importance(feature_importances):
    """
    Plot feature importance
    
    Parameters:
    -----------
    feature_importances : dict
        Dictionary of feature importances
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    """
    # Sort feature importances
    sorted_importances = sorted(
        feature_importances.items(), 
        key=lambda x: x[1],
        reverse=True
    )
    
    # Extract features and importances
    features = [item[0] for item in sorted_importances]
    importances = [item[1] for item in sorted_importances]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix for classification results
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : array-like
        Names of the classes
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        cmap='Blues', 
        fmt='d',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    return fig

def plot_actual_vs_predicted(y_true, y_pred):
    """
    Plot actual vs predicted values for regression results
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    """
    # Convert to numpy arrays if they're not already
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot scatter plot
    scatter = ax.scatter(y_true_np, y_pred_np, alpha=0.6, c='blue', edgecolors='k', label='Data points')
    
    # Plot perfect prediction line (diagonal)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect predictions')
    
    # Add a trendline to show the actual model's predictions
    # This will show whether the model is linear or non-linear
    try:
        # Sort the data to make the trendline smooth
        sort_idx = np.argsort(y_true_np)
        y_true_sorted = y_true_np[sort_idx]
        y_pred_sorted = y_pred_np[sort_idx]
        
        # Use polynomial fitting to show the trend
        z = np.polyfit(y_true_sorted, y_pred_sorted, 3)
        p = np.poly1d(z)
        
        # Add the trendline
        x_trend = np.linspace(min(y_true_np), max(y_true_np), 100)
        y_trend = p(x_trend)
        ax.plot(x_trend, y_trend, 'r-', linewidth=2, label='Model trend')
    except:
        # Fallback if the above fails
        pass
    
    # Label outliers
    residuals = np.abs(y_pred_np - y_true_np)
    threshold = np.std(residuals) * 2
    outliers = residuals > threshold
    
    if np.any(outliers):
        ax.scatter(
            y_true_np[outliers], 
            y_pred_np[outliers], 
            s=80, 
            facecolors='none', 
            edgecolors='red',
            label='Outliers'
        )
    
    # Add a horizontal and vertical grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add labels
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title('Actual vs Predicted Values', fontsize=14)
    
    # Add legend
    ax.legend(loc='best')
    
    # Add a text box with R² score if available
    try:
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        textstr = f'R² = {r2:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
    except:
        pass
    
    plt.tight_layout()
    
    return fig

def plot_clustering(X, cluster_labels):
    """
    Plot clustering results (only for 2D data)
    
    Parameters:
    -----------
    X : pandas.DataFrame
        The features (must be 2D)
    cluster_labels : array-like
        Cluster labels
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Convert to numpy arrays if needed
    X_arr = X.values if hasattr(X, 'values') else X
    
    # Get the first two dimensions
    x1 = X_arr[:, 0]
    x2 = X_arr[:, 1] if X_arr.shape[1] > 1 else np.zeros_like(x1)
    
    # Plot scatter plot with colored clusters
    scatter = ax.scatter(x1, x2, c=cluster_labels, cmap='viridis', alpha=0.7)
    
    # Add legend
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Clustering Results')
    
    plt.tight_layout()
    
    return fig
