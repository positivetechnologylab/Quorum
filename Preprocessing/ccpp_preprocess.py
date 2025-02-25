import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle

def preprocess_ccpp(file_path, save_output=False, output_dir='preprocessed_data'):
    """
    Preprocess/Normalize the Combined Cycle Power Plant (CCPP) dataset.
    
    Args:
    file_path (str): Path to the input CSV file.
    save_output (bool): Whether to save the preprocessed data and metadata.
    output_dir (str): Directory to save the preprocessed data and metadata.
    
    Returns:
    pd.DataFrame: The normalized data.
    list: Indices of anomaly data points.
    dict: Mapping from original indices to preprocessed indices.
    """
    data = pd.read_csv(file_path)
    expected_columns = ['AT', 'V', 'AP', 'RH', 'PE', 'anomaly']
    if not all(col in data.columns for col in expected_columns):
        data = pd.read_csv(file_path, header=None, names=expected_columns)
    
    labels = data['anomaly']
    features = data.drop('anomaly', axis=1)
    
    features = features.astype(float)
    
    anomaly_indices = labels[labels == 'o'].index.tolist()
    
    # normalize numerical features
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    normalized_data = range_based_normalize(scaled_features)
    
    # Create mapping from original to preprocessed indices
    original_to_preprocessed = dict(zip(data.index, normalized_data.index))
    
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        normalized_data.to_csv(os.path.join(output_dir, f'preprocessed_{base_name}.csv'), index=False, header=False)
        
        np.save(os.path.join(output_dir, f'{base_name}_anomaly_indices.npy'), anomaly_indices)
        
        with open(os.path.join(output_dir, f'{base_name}_original_to_preprocessed.pkl'), 'wb') as f:
            pickle.dump(original_to_preprocessed, f)
        
        print(f"Preprocessed data, anomaly indices, and original to preprocessed index mapping saved to {output_dir}")
    
    return normalized_data, anomaly_indices, original_to_preprocessed

def range_based_normalize(data):
    """
    Normalize the data using range-based normalization.
    
    Args:
    data (pd.DataFrame): The input data to normalize.
    
    Returns:
    pd.DataFrame: The normalized data.
    """
    num_features = data.shape[1]
    max_value = 1 / num_features
    normalized_data = pd.DataFrame()
    
    for column in data.columns:
        min_val = data[column].min()
        max_val = data[column].max()
        normalized_data[column] = ((data[column] - min_val) / (max_val - min_val)) * max_value
    
    return normalized_data
