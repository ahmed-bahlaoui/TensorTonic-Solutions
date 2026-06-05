import numpy as np

def angle_features(angles):
    """ Returns: np.ndarray of shape (3, n), rows are sin, cos, tan."""
    angles = np.asarray(angles, dtype=np.float64)
    sin = np.sin(angles)
    cos = np.cos(angles)
    tan = np.tan(angles)
    return np.array([sin, cos, tan])
