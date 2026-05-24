import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    # Your code here
    d_model = np.shape(x)[-1]
    mean = np.mean(x, axis=-1, keepdims=True)
    # variance = (1/d_model) * np.sum((x-mean)**2)
    variance = np.var(x, axis=-1, keepdims=True)
    x_normalized = (x-mean) / np.sqrt(variance + eps)
    x_transformed = gamma * x_normalized + beta

    return x_transformed