import numpy as np

def linear_regression_gradient(X, y, w):
    """
    Returns: dict with 'loss' (float), 'analytical_gradient', 'numerical_gradient' (lists of floats)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n, d = X.shape
        
    
    def compute_loss(X,w,y):
        r = X @ w  - y
        return np.sum( r**2)


    def compute_analytical_gradient(X,y,w):
        r = X@w - y
        return 2 * X.T @ r

    def compute_numerical_gradient(loss_fn, w, h=1e-5):
        grad = np.zeros_like(w)
        
        for j in range(len(w)):
            w_plus = w.copy()
            w_minus = w.copy()
    
            w_plus[j] += h
            w_minus[j] -= h
            grad[j] = (loss_fn(w_plus) - loss_fn(w_minus)) / (2 * h)
    
        return grad
      
    loss_fn = lambda current_w: compute_loss(X, current_w, y)
    

    return {
        "loss": float(compute_loss(X, w, y)),
        "analytical_gradient": compute_analytical_gradient(X, y, w).tolist(),
        "numerical_gradient": compute_numerical_gradient(loss_fn, w).tolist()
    }
        
    
        
