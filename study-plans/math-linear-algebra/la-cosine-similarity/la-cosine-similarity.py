import numpy as np

def cosine_similarity(a, b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    l2_norm_a = np.linalg.norm(a)
    l2_norm_b = np.linalg.norm(b)
    if (l2_norm_a == 0 or l2_norm_b == 0):
        return 0.0

    return (a @ b) / (l2_norm_a * l2_norm_b)