import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    d_model = np.shape(x)[-1]
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_normalized = (x-mean) / np.sqrt(var + eps)
    x_transformed = gamma * x_normalized + beta
    return x_transformed

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v

    head_dim = int(Q.shape[-1] // num_heads)
    Q_reshaped = np.reshape(Q, (Q.shape[0], Q.shape[1], num_heads, head_dim))
    Q_split = Q_reshaped.transpose((0, 2, 1, 3))

    K_reshaped = np.reshape(K, (K.shape[0], K.shape[1], num_heads, head_dim))
    K_split = K_reshaped.transpose((0, 2, 1, 3))

    V_reshaped = np.reshape(V, (V.shape[0], V.shape[1], num_heads, head_dim))
    V_split = V_reshaped.transpose((0, 2, 1, 3))
    
    K_flipped = K_split.transpose(0, 1, 3, 2)

    score = np.matmul(Q_split, K_flipped)
    scores = score / np.sqrt(head_dim)
    scaled = softmax(scores)
    weighted = np.matmul(scaled, V_split)
    V_swapped = weighted.transpose((0, 2, 1, 3))
    merged = V_swapped.reshape(V_swapped.shape[0], V_swapped.shape[1], -1)
    output = merged @ W_o

    return output
    
def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    return np.maximum(0, x@W1 + b1) @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    
    attn_out = multi_head_attention(
        Q=x, K=x, V=x, 
        W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o, 
        num_heads=num_heads
    )

    x_ln1 = layer_norm(x + attn_out, gamma1, beta1)
    ffn_out = feed_forward(x_ln1, W1, b1, W2, b2)
    out = layer_norm(x_ln1 + ffn_out, gamma2, beta2)
    
    return out
    