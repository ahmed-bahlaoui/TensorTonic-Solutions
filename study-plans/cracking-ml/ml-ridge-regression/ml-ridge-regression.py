def ridge_regression(X, y, lr, epochs, alpha):
    """
    Perform ridge regression using gradient descent.
    Returns: tuple of (weights_list, bias)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    n, d = X.shape
    w = np.zeros(d)
    b = float(0.0)

    for _ in range(epochs):
        y_hat = X @ w + b
        dw = 2 * alpha * w +  (2 / n) * (X.T @ (y_hat - y))
        db = (2 / n) * np.sum(y_hat - y)

        w -= lr * dw
        b -= lr * db
    
    return (w, b)
    