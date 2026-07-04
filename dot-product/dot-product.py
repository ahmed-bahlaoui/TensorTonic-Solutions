import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here
    if len(x) != len(y):
        raise ValueError("mismatched lenghts!")
    s = 0
    for a, b in zip(x,y):
        s += a*b

    return s