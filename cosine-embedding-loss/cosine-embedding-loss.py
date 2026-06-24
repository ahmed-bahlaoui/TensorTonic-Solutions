import numpy as np
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    def cos_similarity(x: np.ndarray, y: np.ndarray):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    def compute_loss(x: np.ndarray, y: np.ndarray, label, margin):
        if label == 1.0:
            return 1 - cos_similarity(x,y)
        else:
            return max(0, cos_similarity(x,y) - margin)
        

    return compute_loss(x1,x2, label, margin)
        