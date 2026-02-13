import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    # Write code here
    X = np.array(X, dtype=float)
    
    x_min = X.min(axis=axis, keepdims=True)
    x_max = X.max(axis=axis, keepdims=True)
    
    return (X - x_min) / (x_max - x_min + eps)
