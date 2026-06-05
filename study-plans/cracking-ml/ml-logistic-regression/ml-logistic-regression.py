import numpy as np

def sigmoid(z: np.ndarray):
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1.0/(1.0+np.exp(-z)), np.exp(z)/(1.0+np.exp(z)))
    
    

def logistic_regression(X, y, lr=0.01, n_iters=1000):

    """Returns 
    (weights: np.ndarray, bias: float)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    while n_iters > 0:
        ## Computing the predictions
        z = X @ w + b
        y_hat = sigmoid(z)

        ## Computing the update rules
        dw = (X.T @ (y_hat - y)) / n
        db = np.sum((y_hat - y)) / n

        ## Updating the weights and bias by the update rule
        w -= lr * dw
        b -= lr * db
       
        n_iters -= 1

    return (w,b)
          
          

                            
    
