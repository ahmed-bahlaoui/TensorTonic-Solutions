import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    anchor = np.asarray(anchor, dtype=np.float64)
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)

    positive_distance = np.sum((anchor - positive) ** 2, axis=-1)
    negative_distance = np.sum((anchor - negative) ** 2, axis=-1)
    L = np.maximum(0, positive_distance - negative_distance + margin)
    return float(np.mean(L))