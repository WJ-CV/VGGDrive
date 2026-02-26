import numpy as np

def calculate_steering_angle(trajectory_points, wheelbase=2.5):
    """
    Calculates the front wheel steering angle for each point on a trajectory
    using the bicycle model and Menger curvature.

    Args:
        trajectory_points (list of tuples): A list of (x, y) coordinates
                                           representing the trajectory.
        wheelbase (float): The wheelbase L of the vehicle.

    Returns:
        list of floats: A list of steering angles (in radians) for each
                        point where it can be calculated. The first and
                        last points will have None as steering angle.
    """
    if len(trajectory_points) < 3:
        raise ValueError("Trajectory must contain at least 3 points.")

    steering_angles = [None] * len(trajectory_points) # Initialize with None

    for i in range(1, len(trajectory_points) - 1):
        p_prev = np.array(trajectory_points[i-1])
        p_curr = np.array(trajectory_points[i])
        p_next = np.array(trajectory_points[i+1])

        # Side lengths of the triangle formed by the three points
        a = np.linalg.norm(p_curr - p_next)
        b = np.linalg.norm(p_prev - p_next)
        c = np.linalg.norm(p_prev - p_curr)

        # Avoid division by zero if points are coincident
        if a == 0 or b == 0 or c == 0:
            curvature = 0.0 # Or handle as an error/special case
        else:
            # Area of the triangle using Heron's formula or cross product
            # Using Shoelace formula / cross product for robustness with collinear points
            area = 0.5 * np.abs(p_prev[0]*(p_curr[1] - p_next[1]) + \
                                p_curr[0]*(p_next[1] - p_prev[1]) + \
                                p_next[0]*(p_prev[1] - p_curr[1]))

            # Menger curvature: kappa = 4 * Area / (a * b * c)
            # If area is very small (collinear points), curvature is ~0
            if area < 1e-9: # Tolerance for collinearity
                curvature = 0.0
            else:
                # Need to determine the sign of the curvature.
                # The Menger curvature formula gives unsigned curvature.
                # We can infer the sign from the change in yaw angle or
                # by looking at the cross product of (p_curr - p_prev) and (p_next - p_curr).
                # A positive cross product (z-component) indicates a left turn (positive curvature).
                # A negative cross product indicates a right turn (negative curvature).
                # (v1_x * v2_y) - (v1_y * v2_x)
                v1 = p_curr - p_prev
                v2 = p_next - p_curr

                # Ensure vectors are not zero-length
                if np.linalg.norm(v1) < 1e-9 or np.linalg.norm(v2) < 1e-9:
                    curvature_sign = 0.0
                else:
                    # Normalize vectors before cross product for consistent yaw check,
                    # or use the raw vectors if just checking turn direction.
                    # For curvature sign, the direction of turn matters.
                    cross_product_z = v1[0] * v2[1] - v1[1] * v2[0]
                    curvature_sign = np.sign(cross_product_z)

                # If points are collinear, area is 0, curvature_sign might be 0.
                if curvature_sign == 0 and area > 1e-9 : # Check if not truly collinear but cross_product was zero
                    # This can happen if points are almost collinear or due to precision.
                    # Fallback or more robust sign determination might be needed for noisy data.
                    # For now, if area is non-zero but cross_product is zero, means it's effectively straight.
                     curvature_val = (4 * area) / (a * b * c)
                     curvature = 0.0 # Treat as straight if cross product is numerically zero
                elif area == 0.0:
                    curvature = 0.0
                else:
                    curvature_val = (4 * area) / (a * b * c)
                    curvature = curvature_sign * curvature_val


        # Steering angle: delta = atan(L * kappa)
        steering_angle = np.arctan(wheelbase * curvature)
        steering_angles[i] = steering_angle

    return [np.degrees(i) for i in steering_angles if i is not None]
