import numpy as np

def percentiles(x, q):
    """
    Returns: numpy array of percentile values.
    """

    x = np.asarray(x, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    return np.percentile(x, q)
    
