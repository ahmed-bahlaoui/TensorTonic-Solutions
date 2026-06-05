import numpy as np

def normalize(data):
    data = np.asarray(data, dtype=np.float64)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) 
    normalized_data = (data - mean ) / std 
    return normalized_data