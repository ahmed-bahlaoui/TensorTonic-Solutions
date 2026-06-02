import numpy as np

def linear_regression(X, y, lr, epochs):
    
    """
    Returns: tuple (weights, bias)
    """

    ## Initialising parameters
    X = np.array(X)
    b = 0.0
    n, d = X.shape
    w = np.zeros(d)

    while epochs > 0 :

        y_pred = X @ w + b

        ## Computing the gradient -- Update rules
        dw = (2 / n) * X.T @ (y_pred - y)
        db = (2 / n) * np.sum((y_pred - y))
    
        ## Gradient descent
        w = w - lr * dw
        b = b - lr * db

        epochs -= 1
    
    return (w, b)
