import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    params = []
    momentums = []
    velocities = []
    for i in range(len(param)):
    
        m_t = beta1 * m[i] + (1-beta1) * grad[i]
        v_t = beta2 * v[i] + (1-beta2) * grad[i]**2
        m_hat_t = m_t / (1 - beta1**t)
        v_hat_t = v_t / (1 - beta2**t)
        theta_t = param[i] - lr * (m_hat_t / (np.sqrt(v_hat_t) + eps))
        params.append(theta_t)
        momentums.append(m_t)
        velocities.append(v_t)

    return (tuple(params), tuple(momentums), tuple(velocities)) 