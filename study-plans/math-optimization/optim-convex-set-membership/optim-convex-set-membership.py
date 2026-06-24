import numpy as np

def convex_set_membership(A, b, x):
    """
    Returns: dict with 'in_set' (bool) and 'max_violation' (float, rounded to 6 decimals)
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    threshold = 1e-6
    
    r = A@x - b
    max_violation = np.max(r)

    return {
        "in_set": True if max_violation <= threshold else False,
        "max_violation": max_violation
    }
