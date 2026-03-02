import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    transpose = [list(row) for row in zip(*A)]
    return np.array(transpose)