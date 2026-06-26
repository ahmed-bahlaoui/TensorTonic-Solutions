from math import comb

def binomial_distribution(n, p, threshold):
    """
    Returns: dict with 'pmf' (list), 'mean', 'variance', 'tail_prob' as floats.
    """
    def calc_pmf(k):
        return comb(n, k) * p**k * (1-p)**(n-k)

    pmf = [round(calc_pmf(i),4) for i in range(n + 1)]

    return {
        "pmf": pmf,
        "mean": round(n*p,4),
        "variance": round(n*p*(1-p), 4),
        "prob_at_least": round(sum(pmf[threshold:]), 4)
    }