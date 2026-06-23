import numpy as np

def classify_critical_point(H):
    """
    Returns: one of 'local_min', 'local_max', 'saddle', 'degenerate'
    """
    threshold = 1e-8
    H = np.asarray(H, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(H)
    if np.all(eigenvalues > 0) :
        return "local_min"
    elif np.all(eigenvalues < 0):
        return "local_max"
    elif np.any(np.abs(eigenvalues) < threshold):
        return "degenerate"
    else:
        return "saddle"
    
