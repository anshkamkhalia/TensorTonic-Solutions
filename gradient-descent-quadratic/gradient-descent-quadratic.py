def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    for step in range(steps):
        x = float(x0)
        for _ in range(steps):
            grad = 2*a*x+b
            x = x - (lr * grad)

    return x