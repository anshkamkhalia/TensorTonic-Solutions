import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # Your code here
    relu_input = x @ W1 + b1
    relu_output = np.maximum(0, relu_input)
    return relu_output @ W2 + b2