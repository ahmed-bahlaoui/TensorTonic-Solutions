import numpy as np

def reshape_array(data, operation):
    """
    Returns: ndarray of float64 with shape determined by the operation
    """
    if operation == "transpose":
        return np.array(data, dtype=np.float64).T
    elif operation == "flatten":
        return np.array(data, dtype=np.float64).flatten()
    else:
        y = np.array(data, dtype=np.float64)
        n, m = y.shape
        return y.reshape(1, n, m)
        
