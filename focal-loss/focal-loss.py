import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y ,dtype=np.float64)
    
    f_loss = (1-p)**gamma * (y*np.log(p)) + p**gamma * (1-y)*np.log(1-p)
    f_loss *= -1
    return f_loss.mean()