import numpy as np

def activation_functions(x, activation):
    """
    Returns: list
    """
    ## x  = np.asarray(x, dtype=np.float64)
    pi = np.pi 
    K  = 0.044715
    alpha = 0.01
    
    def sigmoid(z: np.ndarray):
        return 1 / (1 + np.exp(-z))

    def tanh(z: np.ndarray):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def swish(z: np.ndarray): 
        return z * sigmoid(z)

    def g(z: np.ndarray):
        return np.sqrt(2 / pi) * (z + K*z**3)
    
    def GELU(z: np.ndarray):
        return 0.5 * z * (1 + tanh(g(z)))

    def dGELU(z: np.ndarray):
        gz = g(z)
        tanh_gz = np.tanh(gz)
        dg = np.sqrt(2 / pi) * (1 + 3 * K * z**2)
        return 0.5 * (1 + tanh_gz) + 0.5 * z * (1 - tanh_gz**2) * dg

    def RELU(z: np.ndarray):
        return np.where(z >= 0 , z, 0)

    def leaky_RELU(z: np.ndarray):
        return np.where(z > 0 , z, alpha*z)

    
    match activation:
        case "sigmoid":
            return [sigmoid(x), sigmoid(x) * (1 - sigmoid(x))]
        case "relu":
            return np.array([RELU(x), np.where(x > 0 , 1, 0)], dtype=np.float64)

        case "leaky_relu":
            return np.array([leaky_RELU(x), np.where(x > 0 , 1, alpha)], dtype=np.float64)

        case "tanh":
            return [tanh(x), 1- tanh(x)**2]

        case "gelu":
            return [GELU(x), dGELU(x)]

        case "swish":
            return [swish(x), sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)) ]
        
        case _:
            pass