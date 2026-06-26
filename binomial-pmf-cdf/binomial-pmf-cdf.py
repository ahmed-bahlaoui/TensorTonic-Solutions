import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    def pmf_calc(i):
        return comb(n, i) * p**i * (1-p)**(n-i)

    pmf_value = pmf_calc(k)

    cdf_value = 0
    for i in range(k + 1):
        cdf_value += pmf_calc(i)

    return (pmf_value, cdf_value)
