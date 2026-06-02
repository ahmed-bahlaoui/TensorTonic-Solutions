import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    if x == 0:
        return 0.5
    
    y = np.array(x)

    return 1 / (1 + np.exp(-y))
    
