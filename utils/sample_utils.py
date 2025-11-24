import numpy as np
def sample_by_type(data_type_with_index, max_samples_per_type):
    """    
    Parameters:
        - data: numpy array of shape (n, 2) where each row is [type, idx]
        - max_samples_per_type: dict specifying the max sample limit for each type
        
    Returns:
        - A numpy array of sampled [type, idx] pairs
    """
    sampled_data = []
    
    if len(data_type_with_index) == 0:
        return np.zeros((0, 2), dtype=int)

    # Group indices by their types
    unique_types = np.unique(data_type_with_index[:, 0])
    for type_ in unique_types:
        type_indices = data_type_with_index[data_type_with_index[:, 0] == type_]
        
        max_samples = min(len(type_indices), max_samples_per_type.get(type_, len(type_indices)))
        sampled_indices = type_indices[np.random.choice(len(type_indices), max_samples, replace=False)]
        sampled_data.extend(sampled_indices)
    
    return np.asarray(sampled_data).reshape(-1, 2)
