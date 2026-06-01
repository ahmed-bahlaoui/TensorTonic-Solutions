import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    categories, counts = np.unique(y, return_counts=True)
    if len(y) == 0 or len (categories) == 1:
        return float(0)
    ## We should calculate the proportion for each category
    p = np.array([i / len(y) for i in counts])
    logp = np.log2(p)
    res = -1 * np.sum(p*logp)
    return float(res)  
