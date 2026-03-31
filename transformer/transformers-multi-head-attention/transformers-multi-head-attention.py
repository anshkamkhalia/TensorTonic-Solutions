import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    q_proj = Q @ W_q
    k_proj = K @ W_k
    v_proj = V @ W_v

    def split_heads(x):
        x = x.reshape(batch_size, seq_len, num_heads, d_k)
        return x.transpose(0, 2, 1, 3)
    
    q_heads = split_heads(q_proj) 
    k_heads = split_heads(k_proj) 
    v_heads = split_heads(v_proj) 
    
    scaled_dot_product = (q_heads @ k_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    weights = softmax(scaled_dot_product, axis=-1)
    context_heads = weights @ v_heads    
    concat = context_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    output = concat @ W_o
    
    return output