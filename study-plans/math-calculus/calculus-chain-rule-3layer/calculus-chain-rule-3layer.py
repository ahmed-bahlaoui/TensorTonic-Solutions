import numpy as np

def chain_rule_3layer(w1, w2, w3, x):
    """
    Returns: dict with 'factors' (list of 6 floats),
    'analytical_gradient' (float),
    'numerical_gradient' (float)
    """

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def dsigmoid(z):
        s = sigmoid(z)
        return s * (1 - s)

    # Forward pass
    z1 = w1 * x
    a1 = sigmoid(z1)

    z2 = w2 * a1
    a2 = sigmoid(z2)

    z3 = w3 * a2
    y = sigmoid(z3)

    # Chain rule factors:
    # dy/dw1 = dy/dz3 * dz3/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/dw1

    dy_dz3 = dsigmoid(z3)
    dz3_da2 = w3
    da2_dz2 = dsigmoid(z2)
    dz2_da1 = w2
    da1_dz1 = dsigmoid(z1)
    dz1_dw1 = x

    factors = [
        dy_dz3,
        dz3_da2,
        da2_dz2,
        dz2_da1,
        da1_dz1,
        dz1_dw1
    ]

    analytical_gradient = (
        dy_dz3
        * dz3_da2
        * da2_dz2
        * dz2_da1
        * da1_dz1
        * dz1_dw1
    )

    # Numerical gradient using finite differences
    eps = 1e-5

    def f(w1_value):
        return sigmoid(w3 * sigmoid(w2 * sigmoid(w1_value * x)))

    numerical_gradient = (f(w1 + eps) - f(w1 - eps)) / (2 * eps)

    return {
        "factors": [float(v) for v in factors],
        "analytical_gradient": float(analytical_gradient),
        "numerical_gradient": float(numerical_gradient)
    }