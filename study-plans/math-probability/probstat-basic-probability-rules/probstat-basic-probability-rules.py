def basic_probability(p_a, p_b, p_a_and_b):
    """
    Returns: [p_union, p_a_complement, p_b_complement, p_a_and_not_b] as a list.
    """
    p_union = p_a + p_b - p_a_and_b
    p_a_complement = 1 - p_a
    p_b_complement = 1 - p_b
    p_a_and_not_b = p_a - p_a_and_b

    a =  [p_union, p_a_complement, p_b_complement, p_a_and_not_b]
    return [round(i, 2) for i in a]
    
    