import numpy as np


def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
        
        
    for _ in range(epochs):
       for i in range(n):
            # 1. Calculate prediction for a SINGLE row
            z = np.dot(X[i], w) + b
            y_pred = 1 if z >= 0 else 0
            
            # 2. Calculate error for that single prediction
            error = y[i] - y_pred
            
            # 3. Update weights and bias IMMEDIATELY
            w += lr * error * X[i]
            b += lr * error

    return w, float(b)