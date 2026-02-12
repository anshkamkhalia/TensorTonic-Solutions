import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    if len(a) == 0 or len(b) == 0:
        return 0
    
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)

    if mag_a == 0 or mag_b == 0:
        return 0

    cosine_similarity = (np.dot(a,b)) / (mag_a * mag_b)
    return cosine_similarity