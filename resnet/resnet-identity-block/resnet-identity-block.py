import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    h = np.maximum(0, x@W1.T)
    F_x = h @ W2.T
    y = np.maximum(0, F_x + x)
    return np.array(y)