import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    expected_future_value = np.dot(transitions, values)
    q_values = rewards + gamma * expected_future_value
    updated_values = np.max(q_values, axis=1)
    
    return list(updated_values)