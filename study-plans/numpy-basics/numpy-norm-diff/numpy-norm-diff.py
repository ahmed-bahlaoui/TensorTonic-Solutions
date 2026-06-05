def norm_diff(a, b, lo, hi):
    """Returns: np.ndarray of absolute differences after clipping and rescaling to [0, 1]"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = np.clip(a, lo, hi)
    b = np.clip(b, lo, hi)

    rescaled_a = (a - lo) / ( hi - lo)
    rescaled_b = (b - lo) / ( hi - lo)

    
    return np.abs(rescaled_a - rescaled_b)