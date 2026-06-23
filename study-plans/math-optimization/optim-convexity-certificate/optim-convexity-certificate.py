import numpy as np

def convexity_certificate(H):
    """
    Returns: dict with 'is_convex' (bool) and 'min_eigenvalue' (float, rounded to 6 decimals)
    """
    threshold = -1e-6
    eigenvalues = np.linalg.eigvalsh(H)
    lambda_min = np.min(eigenvalues)
    return {"is_convex": True if  lambda_min - threshold >= 0 else False,
           "min_eigenvalue": lambda_min}

    
