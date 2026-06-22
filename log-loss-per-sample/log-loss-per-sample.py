import math
import numpy as np	

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    p_hat = np.clip(y_pred, eps, 1-eps)
    log_loss = y_true * np.log(p_hat) + (1- y_true) * np.log(1- p_hat)
    return list(-log_loss)