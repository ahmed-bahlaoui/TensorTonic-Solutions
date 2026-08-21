import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    """Return the coefficient of determination."""
    # Write code here
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    mean = np.mean(y_true)

    if np.array_equal(y_pred, y_true):
        return 1.0

    if np.array_equal(y_true-mean, np.zeros(len(y_true))):
        return 0.0
    

    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum( (y_true-mean)**2 )
    return float( 1 - (numerator/denominator) )
    
    