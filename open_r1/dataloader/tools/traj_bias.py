
import numpy as np
def extend_trajectory_kinematic(initial_trajectory_points, num_points_to_extend=20, num_accel_to_average=3):
    """
    使用运动学模型（恒定加速度）自然地延长3D轨迹。

    扩展是基于从初始轨迹末端计算出的平均加速度。

    参数:
        initial_trajectory_points (list or np.ndarray):
            初始轨迹点。预期是一个列表的列表/元组，或一个 NumPy 数组，
            形状为 (N, 3)，其中 N 是点的数量（例如, 60）。
            每个点是 [x, y, z]。
        num_points_to_extend (int):
            为扩展生成的轨迹点数量（例如, 40）。
        num_accel_to_average (int):
            用于计算恒定外推加速度的初始轨迹末端加速度向量的数量。
            如果可用的加速度数量少于此值，则将对所有可用的加速度进行平均。

    返回:
        np.ndarray:
            一个形状为 (num_points_to_extend, 3) 的 NumPy 数组，
            包含延长后的轨迹点。
    """
    points = np.asarray(initial_trajectory_points, dtype=float)
    n_initial_points = points.shape[0]

    # 处理初始点过少的情况
    if n_initial_points == 0:
        # 如果没有输入点，返回空数组
        return np.empty((0, 3)) 
    
    # 获取初始轨迹的最后一个点，作为延长的起点
    last_point_of_initial_trajectory = points[-1, :].copy()

    if n_initial_points == 1:
        # 如果只有一个点，通过重复该点进行延长（零速度，零加速度）
        return np.array([last_point_of_initial_trajectory] * num_points_to_extend)

    # 计算速度 (假设点之间的时间步长 dt=1)
    # velocities[i] 是从 points[i] 到 points[i+1] 的速度
    velocities = np.diff(points, axis=0)  # 形状: (n_initial_points - 1, 3)
    
    # 用于延长第一步的初始速度 (即原始轨迹的最后一个速度)
    initial_velocity_for_extension = velocities[-1, :].copy()

    # 计算加速度 (假设点之间的时间步长 dt=1)
    if n_initial_points < 3 or velocities.shape[0] < 2:
        # 点数不足以计算加速度 (例如, 初始轨迹只有2个点，因此只有1个速度向量)
        # 在这种情况下，我们假设加速度为零，即匀速延长
        constant_acceleration_for_extension = np.zeros(3)
    else:
        # accelerations[i] 是从 velocities[i] 到 velocities[i+1] 的加速度变化
        accelerations = np.diff(velocities, axis=0)  # 形状: (n_initial_points - 2, 3)
        
        if accelerations.shape[0] == 0: 
            # 这种情况应该被上面的 n_initial_points < 3 覆盖
            constant_acceleration_for_extension = np.zeros(3)
        else:
            # 确定实际用于平均的加速度数量
            num_to_actually_average = min(num_accel_to_average, accelerations.shape[0])
            if num_to_actually_average > 0:
                constant_acceleration_for_extension = np.mean(accelerations[-num_to_actually_average:], axis=0)
            else:
                # 如果 accelerations.shape[0] > 0，则不应发生此情况
                constant_acceleration_for_extension = np.zeros(3)
    
    # --- 外推循环 ---
    extended_points_list = []
    current_point = last_point_of_initial_trajectory  # 当前点初始化为原始轨迹的最后一个点
    current_velocity = initial_velocity_for_extension # 当前速度初始化为原始轨迹的最后一个速度 (作为延长段的初始速度)

    for _ in range(num_points_to_extend):
        # 运动学公式: p_next = p_current + v_current*dt + 0.5*a*(dt)^2
        # 假设 dt=1 (点之间的时间步长)
        # 注意: current_velocity 是当前时间步开始时的速度
        next_point = current_point + current_velocity + 0.5 * constant_acceleration_for_extension
        extended_points_list.append(next_point.copy()) # 使用 .copy() 存储值
        
        # 更新下一个时间步开始时的速度: v_next = v_current + a*dt
        # 假设 dt=1
        next_velocity = current_velocity + constant_acceleration_for_extension
        
        # 为下一次迭代更新当前状态
        current_point = next_point
        current_velocity = next_velocity
        
    return np.array(extended_points_list)


def compute_trajectory_bias(train_val_traj, train_test_split_point, num_points_to_extend, num_accel_to_average):
    extended_trajectory = extend_trajectory_kinematic(train_val_traj[:train_test_split_point], num_points_to_extend, num_accel_to_average)
    from sklearn.metrics import mean_squared_error

    def calculate_metrics(predicted, gt):
        """Calculate various comparison metrics"""
        
        rmse_total = np.sqrt(mean_squared_error(gt, predicted))
        
        return rmse_total

    return calculate_metrics(extended_trajectory, np.array(train_val_traj)[train_test_split_point:]), extended_trajectory.tolist(), train_val_traj[train_test_split_point:]
