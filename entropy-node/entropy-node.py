import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    entropy = 0.0

    if len(y) == 0:
        return entropy
    
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)

    entropy = -np.sum(probs * np.log2(probs))
    
    return float(np.abs(entropy))