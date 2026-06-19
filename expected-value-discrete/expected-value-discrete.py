import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if np.abs(np.sum(p) - 1) >= 1e-6:
        raise ValueError("probabilities do not sum up to 1!")
    elif len(x) != len(p):
        raise ValueError("The array shapes do not match!")
    return np.dot(x, p)
