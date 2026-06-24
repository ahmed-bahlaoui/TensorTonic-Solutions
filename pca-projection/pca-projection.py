import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    ## First, we need to center the data
    def center(X: np.ndarray) -> np.ndarray:
        return X - np.mean(X, axis=0, keepdims=True)

    X_c = center(X)

    ## Computing the covariance matrix
    C = X_c.T @ X_c
    C /= len(X_c)-1

    ## Finding top-k eigenvalues of C
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    top_k_vecs = eigenvectors[:, -k:][:, ::-1]
    X_proj = X_c @ top_k_vecs
    return X_proj