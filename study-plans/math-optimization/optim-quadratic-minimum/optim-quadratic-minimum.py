def quadratic_minimum(a, b, c):
    """
    Returns: dict with 'x_star' and 'f_min' (floats), each rounded to 6 decimals
    """
    x_star = -b / (2*a)
    f_min  = c - b*b / (4*a) 
    return {'x_star': x_star, 'f_min': f_min}
