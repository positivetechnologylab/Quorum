import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

def perform_pca(data):
    """
    Perform PCA on the input data.
    
    Args:
    data (pd.DataFrame): Input data
    
    Returns:
    PCA: Fitted PCA object
    np.ndarray: Transformed data
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)
    
    return pca, pca_data

def select_features(data, num_qubits, strategy='a'):
    """
    Select features based on the specified strategy.
    
    Args:
    data (pd.DataFrame): Input data
    num_qubits (int): Number of qubits specified in main
    strategy (str): Feature selection strategy (a, b, c, d, or e)
    
    Returns:
    pd.DataFrame: Data with selected features (including added 0-features if necessary)
    list: Indices of selected features
    """
    num_features = 2**num_qubits - 1
    original_num_features = data.shape[1]
    
    #if the number of features is greater than the original number of features, add zero features rather than selecting features
    if num_features >= original_num_features:
        selected_data = data.copy()
        num_zero_features = num_features - original_num_features
        for i in range(num_zero_features):
            selected_data[f'zero_feature_{i}'] = 0
        return selected_data, list(range(num_features))
    
    #uniform random selection of features
    if strategy == 'e':
        selected_features = np.random.choice(data.columns, num_features, replace=False)
        return data[selected_features], selected_features.tolist()
    
    pca, pca_data = perform_pca(data)
    feature_importance = np.abs(pca.components_).sum(axis=0)
    
    #Select the top num_features features
    if strategy == 'a':
        selected_indices = feature_importance.argsort()[::-1][:num_features]
    #Select the bottom num_features features
    elif strategy == 'b':
        selected_indices = feature_importance.argsort()[::-1][-num_features:]
    #Select the top half and bottom half of the features
    elif strategy == 'c':
        num_top = num_features // 2
        num_bottom = num_features - num_top
        top_indices = feature_importance.argsort()[::-1][:num_top]
        bottom_indices = feature_importance.argsort()[::-1][-num_bottom:]
        selected_indices = np.concatenate([top_indices, bottom_indices])
    
    #weighted random selection based on feature importance
    elif strategy == 'd':
        selected_indices = np.random.choice(
            len(feature_importance),
            num_features,
            replace=False,
            p=feature_importance / feature_importance.sum()
        )
    else:
        raise ValueError("Invalid strategy. Choose 'a', 'b', 'c', 'd', or 'e'.")
    
    selected_features = data.columns[selected_indices]
    return data[selected_features], selected_indices.tolist()