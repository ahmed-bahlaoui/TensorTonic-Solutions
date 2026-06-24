import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    threshold = 1e-8
    g = np.asarray(g, dtype=np.float64)
    norm_g = np.linalg.norm(g)
    if norm_g <= max_norm or max_norm <= threshold:
        return g
    k = max_norm / norm_g
    return np.asarray(g * k, dtype=np.float64)