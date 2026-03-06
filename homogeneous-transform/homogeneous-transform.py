import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    T = np.asarray(T)
    points = np.asarray(points)
    is_single_point = points.ndim == 1    
    if is_single_point:
        points = points.reshape(1, 3)

    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    transformed_h = points_h @ T.T
    result = transformed_h[:, :3]
    
    return result.flatten() if is_single_point else result