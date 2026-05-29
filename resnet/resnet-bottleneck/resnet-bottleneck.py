import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """

    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    W3 = np.array(W3)
    Ws =  np.array(Ws)

    shortcut = None
    
    y_1 = np.maximum(0, x@W1)
    y_2 = np.maximum(0, y_1@W2)
    y_3 = y_2 @ W3 

    if y_3.shape[-1] == x.shape[-1]:
        shortcut = x
    else:
        shortcut = x @ Ws
        
    out = np.maximum(0, y_3 + shortcut)
    
    return out