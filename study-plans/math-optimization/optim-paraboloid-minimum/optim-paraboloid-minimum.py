def paraboloid_minimum(a, b, c, d, e):
    """
    Returns: dict with 'x_star', 'y_star', 'f_min' (floats), each rounded to 6 decimals
    """
    x_star = -c / (2*a)
    y_star = -d / (2*b)
    def f(x,y):
        return a*x**2 + b*y**2 + c*x + d*y + e
    
    f_min = f(x_star, y_star)

    return {
        "x_star": x_star,
        "y_star": y_star,
        "f_min": f_min
    }
