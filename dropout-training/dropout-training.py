import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)
    
    if p == 1.0:
        return np.zeros_like(x, dtype=float), np.zeros_like(x, dtype=float)

    keep_prob = 1.0 - p
    scale = 1.0 / keep_prob

    if rng is None:
        random_vals = np.random.random(x.shape)
    else:
        random_vals = rng.random(x.shape)

    mask = random_vals < keep_prob
    dropout_pattern = np.where(mask, scale, 0.0)    
    output = x * dropout_pattern

    return output, dropout_pattern
