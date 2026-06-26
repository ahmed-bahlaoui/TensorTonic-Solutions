from math import factorial, exp

def poisson_distribution(lam, max_k):
    """
    Returns: [pmf_list, cdf_at_max_k, p_zero] as a list.
    """
    def compute_poisson(k):
        return lam**k * exp(-lam)/ factorial(k)

    pmf = [round(compute_poisson(i),4) for i in range(max_k + 1)]
    p_up_to_k = round(sum(pmf[:max_k + 1]),4)
    p_of_0 = round(compute_poisson(0),4)

    return [pmf, p_up_to_k, p_of_0]