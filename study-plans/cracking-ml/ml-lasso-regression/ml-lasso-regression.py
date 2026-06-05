def lasso_regression(X, y, lr, epochs, alpha):
    """
    Perform Lasso Regression using gradient descent with L1 subgradient.
    Returns: tuple of (weights_list, bias_float)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    ## Initializing weights and bias
    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        y_hat = X @ w + b
        ## Computing gradients

        dw = (2 / n) * (X.T @ (y_hat - y)) + alpha * np.sign(w)
        db = (2 / n) * np.sum(y_hat - y)
        
        ## Updating weights and biases
        w -= lr * dw
        b -= lr * db

    return (w, b)