import numpy as np

def filter_and_extract(data, row_start, row_stop, threshold):
    """
    Returns: 1D ndarray of float64
    """
    data = np.asarray(data, dtype=np.float64)
    filtered_data =  data[row_start:row_stop]
    mask = filtered_data > threshold
    hadamard = filtered_data * mask
    l = [i for i in hadamard.flatten() if i > 0]
    return np.asarray(l, dtype=np.float64)

        
