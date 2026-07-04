def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    B = lam * np.eye(d)
    inv = np.linalg.inv((X.T@X + B))
    w =  inv @ X.T @ y
    return w