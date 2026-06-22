import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.asarray(x, dtype=np.float64)

    if np.ndim(x) == 1:
        numerator = np.exp(x - np.max(x))
        return numerator / np.sum(numerator)
    else:
        # Subtract the maximum value in each row for numerical stability
        max_per_row = np.max(x, axis=1, keepdims=True)
        numerator = np.exp(x - max_per_row)
        
        # Sum along each row to get the denominator
        denominator = np.sum(numerator, axis=1, keepdims=True)
        
        return numerator / denominator