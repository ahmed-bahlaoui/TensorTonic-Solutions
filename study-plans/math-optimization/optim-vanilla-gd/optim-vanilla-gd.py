import numpy as np

def vanilla_gradient_descent(x0, y0, lr, n_iters):
    """
    Returns: dict with 'trajectory' (list of [x,y] pairs), 'final_point' ([x,y]), 'final_value' (float)
    """
    x = float(x0)
    y = float(y0)
    

    def func(x,y):
        return x*x + 3*y*y

    d = {
        "trajectory": [[x, y]],
        "final_point": [x,y],
        "final_value": func(x,y)
    }
    
    for _ in range(n_iters):

        ## Updating gradient after each run
        
        dfdx = 2 * x
        dfdy = 6 * y

        ## Updating coordinates after each run
        x = x - lr * dfdx
        y = y - lr * dfdy

        #Appending to array
        d.get("trajectory").append([x,y])
        
    d["final_point"] = [x,y]
    
    d["final_value"] = func(x,y)

    return d