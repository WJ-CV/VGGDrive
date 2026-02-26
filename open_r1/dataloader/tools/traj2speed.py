import math

def calculate_velocities(trajectory, time_interval):
    """
    Calculates the velocity at each interior point of a trajectory.

    Args:
        trajectory: A list of (x, y) tuples representing the points in the trajectory.
        time_interval: The time interval between consecutive points in seconds.

    Returns:
        A list of velocity magnitudes (m/s) for each interior point.
        Returns an empty list if the trajectory has fewer than 3 points.
    """
    velocities = []
    if len(trajectory) < 3:
        print("Trajectory must have at least 3 points to calculate interior velocities.")
        return velocities

    for i in range(1, len(trajectory) - 1):
        x_prev, y_prev = trajectory[i-1]
        x_curr, y_curr = trajectory[i] # Current point, not directly used in central difference for position
        x_next, y_next = trajectory[i+1]

        # Calculate delta x and delta y using central difference
        delta_x = x_next - x_prev
        delta_y = y_next - y_prev

        # Calculate velocity components
        vx = delta_x / (2 * time_interval)
        vy = delta_y / (2 * time_interval)

        # Calculate velocity magnitude
        velocity_magnitude = math.sqrt(vx**2 + vy**2)
        velocities.append(velocity_magnitude)

    return velocities