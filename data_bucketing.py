import numpy as np
from typing import List, Tuple

def estimate_bucket_size(p_anomaly: float, target_probability: float, tolerance: float = 1e-6, max_iterations: int = 1000) -> int:
    """
    Estimate the bucket size needed to achieve a target probability
    of containing at least one anomaly.
    
    Args:
    p_anomaly (float): Probability of an anomaly in a single sample
    target_probability (float): Desired probability of having at least one anomaly in a bucket
    tolerance (float): Acceptable error in the probability (default: 1e-6)
    max_iterations (int): Maximum number of iterations (default: 1000)
    
    Returns:
    int: Estimated bucket size needed
    """
    n = 1
    
    for _ in range(max_iterations):
        current_P = 1 - (1 - p_anomaly) ** n
        
        if abs(current_P - target_probability) < tolerance:
            return n
        
        if current_P < target_probability:
            n += 1
        else:
            return n  #round up to ensure we meet/exceed the target probability
    
    raise ValueError(f"Failed to converge after {max_iterations} iterations")

def create_data_buckets(num_datapoints: int, num_anomalies: int, target_probability: float = 0.5) -> List[List[int]]:
    """
    Create buckets of random indices for the dataset.
    
    Args:
    num_datapoints (int): Total number of datapoints in the dataset
    num_anomalies (int): Total number of anomalies in the dataset
    target_probability (float): Desired probability of having at least one anomaly in a bucket
    
    Returns:
    List[List[int]]: List of buckets, where each bucket is a list of indices
    """
    p_anomaly = num_anomalies / num_datapoints
    bucket_size = estimate_bucket_size(p_anomaly, target_probability)
    
    all_indices = list(range(num_datapoints))
    np.random.shuffle(all_indices)
    
    buckets = [all_indices[i:i+bucket_size] for i in range(0, num_datapoints, bucket_size)]
    
    return buckets

def perform_bucketing(preprocessed_data: np.ndarray, high_risk_indices: List[int], target_probability: float = 0.5) -> Tuple[List[List[int]], int]:
    """
    Perform the bucketing process on the preprocessed data.
    
    Args:
    preprocessed_data (np.ndarray): The preprocessed dataset
    high_risk_indices (List[int]): List of indices of high-risk (anomalous) datapoints
    target_probability (float): Desired probability of having at least one anomaly in a bucket
    
    Returns:
    Tuple[List[List[int]], int]: A tuple containing the list of buckets and the bucket size
    """
    num_datapoints = len(preprocessed_data)
    num_anomalies = len(high_risk_indices)
    
    buckets = create_data_buckets(num_datapoints, num_anomalies, target_probability)
    bucket_size = len(buckets[0])  # all buckets except possibly the last one will have this size
    
    print(f"Created {len(buckets)} buckets with a target size of {bucket_size} datapoints each.")
    print(f"Probability of at least one anomaly in each bucket: {target_probability}")
    
    return buckets, bucket_size