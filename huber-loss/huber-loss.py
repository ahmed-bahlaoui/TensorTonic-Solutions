import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    error = np.abs(y_true - y_pred)
    
    quadratic = 0.5 * error**2
    linear = delta * (error - 0.5 * delta)
    
    loss = np.where(error <= delta, quadratic, linear)
  
    return np.mean(loss)