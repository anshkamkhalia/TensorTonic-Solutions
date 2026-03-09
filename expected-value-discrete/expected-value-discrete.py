import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    if sum(p) != 1:
        raise ValueError
    
    expected_value = 0
    for x_val, prob in zip(x,p):
        expected_value += x_val * prob

    return expected_value
