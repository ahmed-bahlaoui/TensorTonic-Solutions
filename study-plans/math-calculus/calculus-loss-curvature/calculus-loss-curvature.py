import numpy as np

def loss_curvature_analysis(y_hat, y, delta):
    """
    Returns: dict with 'mse', 'ce', 'huber' keys,
    each containing 'dL' and 'd2L' lists
    """

    y_hat = np.asarray(y_hat, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # avoid division by zero for cross-entropy
    eps = 1e-12
    y_hat_safe = np.clip(y_hat, eps, 1 - eps)

    error = y_hat - y

    # MSE
    dL_mse = 2 * error
    d2L_mse = np.full_like(y_hat, 2.0, dtype=np.float64)

    # Binary cross-entropy
    dL_ce = (y_hat_safe - y) / (y_hat_safe * (1 - y_hat_safe))
    d2L_ce = y / (y_hat_safe ** 2) + (1 - y) / ((1 - y_hat_safe) ** 2)

    # Huber loss
    abs_error = np.abs(error)

    dL_huber = np.where(
        abs_error <= delta,
        error,
        delta * np.sign(error)
    )

    d2L_huber = np.where(
        abs_error <= delta,
        1.0,
        0.0
    )

    return {
        "mse": {
            "dL": dL_mse.tolist(),
            "d2L": d2L_mse.tolist()
        },
        "ce": {
            "dL": dL_ce.tolist(),
            "d2L": d2L_ce.tolist()
        },
        "huber": {
            "dL": dL_huber.tolist(),
            "d2L": d2L_huber.tolist()
        }
    }