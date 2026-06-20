import numpy as np

def activation_derivative(name, x):
    """
    Returns: list of floats (the derivative evaluated at each x)
    """
    x = np.asarray(x, dtype=np.float64)
    def sigmoid(z: np.ndarray):
        return 1 / (1 + np.exp(-z))

    def reLU(z : np.ndarray):
        return np.where(z >= 0, z, 0)

    
    def tanh(z: np.ndarray):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def swish(z: np.ndarray):
        return z * sigmoid(z)
        
    match name:
        case 'sigmoid':
            return sigmoid(x) * (1 - sigmoid(x))
        case "relu":
            return np.where(x > 0, 1, 0)
        case "swish":
            return sigmoid(x) + x *sigmoid(x) * (1 - sigmoid(x))
        case "tanh":
            return 1 - tanh(x)*tanh(x)
        case _:
            raise ValueError("not a valid activation function!")
            
