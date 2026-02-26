import numpy as np

def calculate_steering_angle(trajectory_points, wheelbase=2.5, min_segment_length=5e-1, min_triangle_denominator=1e-3):
    """
    Calculates the front wheel steering angle for each point on a trajectory
    using the bicycle model and Menger curvature, with improvements for stability
    when points are close.

    Args:
        trajectory_points (list of tuples): A list of (x, y) coordinates
                                            representing the trajectory.
        wheelbase (float): The wheelbase L of the vehicle.
        min_segment_length (float): Minimum length for segments (p_prev-p_curr, p_curr-p_next)
                                    to be considered for reliable curvature calculation.
                                    If segment lengths 'a' or 'c' are below this,
                                    curvature is treated as 0. Default: 0.001 (1mm).
        min_triangle_denominator (float): Minimum value for the denominator (a*b*c) in
                                          Menger curvature calculation to avoid instability.
                                          If (a*b*c) is below this, curvature is treated as 0.
                                          Default: 1e-9.

    Returns:
        list: A list of steering angles (in degrees) for each point.
              The first and last points will have None as their steering angle.
              Calculated steering angles for very short segments or degenerate
              triangles will be 0.0 degrees.
    """
    if len(trajectory_points) < 3:
        raise ValueError("Trajectory must contain at least 3 points.")

    steering_angles_rad = [None] * len(trajectory_points) # Initialize with None for radians

    for i in range(1, len(trajectory_points) - 1):
        p_prev = np.array(trajectory_points[i-1])
        p_curr = np.array(trajectory_points[i])
        p_next = np.array(trajectory_points[i+1])

        # Side lengths of the triangle formed by the three points
        # c = distance(prev, curr)
        # a = distance(curr, next)
        # b = distance(prev, next)
        c_dist = np.linalg.norm(p_prev - p_curr)
        a_dist = np.linalg.norm(p_curr - p_next)

        # If segments 'a' or 'c' are too short, reliable curvature calculation is difficult.
        # Assume straight path (curvature = 0) for such cases.
        if a_dist < min_segment_length or c_dist < min_segment_length:
            steering_angles_rad[i] = 0.0
            continue

        b_dist = np.linalg.norm(p_prev - p_next)

        # Handle cases where points might be coincident or form a degenerate triangle,
        # though min_segment_length should catch most issues with a_dist and c_dist.
        if a_dist == 0 or b_dist == 0 or c_dist == 0: # Should be rare if min_segment_length > 0
            curvature = 0.0
        else:
            # Area of the triangle using Shoelace formula (or cross product magnitude)
            area = 0.5 * np.abs(p_prev[0]*(p_curr[1] - p_next[1]) + \
                                p_curr[0]*(p_next[1] - p_prev[1]) + \
                                p_next[0]*(p_prev[1] - p_curr[1]))

            denominator = a_dist * b_dist * c_dist

            # If area is very small (collinear points) or denominator is too small (instability)
            if area < 1e-4 or denominator < min_triangle_denominator:
                curvature = 0.0
            else:
                # Menger curvature (unsigned): kappa_unsigned = 4 * Area / (a * b * c)
                curvature_val = (4 * area) / denominator

                # Determine the sign of the curvature
                v1 = p_curr - p_prev # Vector from prev to curr
                v2 = p_next - p_curr # Vector from curr to next

                # Cross product's z-component determines turn direction
                # (v1_x * v2_y) - (v1_y * v2_x)
                # A positive cross product indicates a left turn (positive curvature).
                # A negative cross product indicates a right turn (negative curvature).
                cross_product_z = v1[0] * v2[1] - v1[1] * v2[0]
                curvature_sign = np.sign(cross_product_z)

                # If points are collinear, area and cross_product_z will be near zero.
                # If numerically cross_product_z is zero but area suggests non-collinearity (unlikely with checks),
                # or if truly collinear, curvature should be zero.
                if curvature_sign == 0:
                    curvature = 0.0
                else:
                    curvature = curvature_sign * curvature_val

        # Steering angle: delta = atan(L * kappa)
        steering_angle_rad = np.arctan(wheelbase * curvature)
        steering_angles_rad[i] = steering_angle_rad

    # Convert calculated angles to degrees, keep None for first/last
    steering_angles_deg = []
    for angle_rad in steering_angles_rad:
        if angle_rad is not None:
            steering_angles_deg.append(np.degrees(angle_rad))
    return steering_angles_deg