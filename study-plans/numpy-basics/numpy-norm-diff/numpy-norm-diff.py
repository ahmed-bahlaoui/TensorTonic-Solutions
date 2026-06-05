def norm_diff(a, b, lo, hi):
    """Returns: np.ndarray of absolute differences after clipping and rescaling to [0, 1]"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = np.clip(a, lo, hi)
    b = np.clip(b, lo, hi)

    def rescale(x):
        return  (x - lo) / (hi - lo)

    
    return np.abs(rescale(a) - rescale(b))