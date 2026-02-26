import math

def distance(p1, p2):
    """计算两点之间的欧几里得距离"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_curvature_at_point(p_prev, p_curr, p_next):
    """
    根据三个连续点计算中间点的曲率 (kappa = 1/R).
    使用 Menger 曲率公式: kappa = 4 * Area / (a*b*c),
    其中 a, b, c 是由三点构成的三角形的边长。
    p_prev, p_curr, p_next 分别是前一个点、当前点、后一个点。
    """
    # 三角形的边长
    # side_c 是 p_prev 到 p_curr 的距离
    # side_a 是 p_curr 到 p_next 的距离
    # side_b 是 p_prev 到 p_next 的距离 (弦长)
    side_c = distance(p_prev, p_curr)
    side_a = distance(p_curr, p_next)
    side_b = distance(p_prev, p_next)

    epsilon = 1e-9  # 用于浮点数比较的小量

    # 如果任何一段的长度过小，或者点重合，则曲率视为0
    if side_a < epsilon or side_c < epsilon:
        return 0.0

    # 使用海伦公式或坐标行列式计算三角形面积
    # Area = 0.5 * |x_prev(y_curr - y_next) + x_curr(y_next - y_prev) + x_next(y_prev - y_curr)|
    area = 0.5 * abs(
        p_prev[0] * (p_curr[1] - p_next[1]) +
        p_curr[0] * (p_next[1] - p_prev[1]) +
        p_next[0] * (p_prev[1] - p_curr[1])
    )

    # 如果面积过小（点共线），则曲率视为0
    if area < epsilon:
        return 0.0

    # 曲率公式的分母
    denominator = side_a * side_b * side_c
    if denominator < epsilon: # 应该已被前面的检查捕获
        return 0.0

    # 曲率 kappa = 1/R = (4 * Area) / (side_a * side_b * side_c)
    kappa = (4 * area) / denominator
    return kappa

def calculate_trajectory_dynamic_strain_index(trajectory_points):
    """
    计算轨迹的动态应变指数。
    trajectory_points: 一个包含 (x, y) 坐标元组的列表。
    """
    num_points = len(trajectory_points)
    if num_points < 3:
        # 需要至少三个点来计算曲率
        return 0.0

    total_metric = 0.0
    epsilon = 1e-9

    # 遍历轨迹中的内部点 (从第二个点到倒数第二个点)
    # 对于10个点 (索引 0-9), i 的范围是 1 到 8
    for i in range(1, num_points - 1):
        p_prev = trajectory_points[i-1]
        p_curr = trajectory_points[i]
        p_next = trajectory_points[i+1]

        L_prev = distance(p_prev, p_curr)
        L_next = distance(p_curr, p_next)

        # 如果当前点与前后点距离都极小（例如车辆停止或数据点密集重合）
        # S_i 会很小，其贡献也会很小，这是合理的。
        # 若L_prev, L_next之一为0，可能说明点有重复，但calculate_curvature_at_point会处理。

        kappa_i = calculate_curvature_at_point(p_prev, p_curr, p_next)

        # 速度代理 S_i: 与点 P_i 相关联的平均线段长度
        # S_i 是一个长度，代表 v * delta_t (速度 * 时间间隔)
        S_i = (L_prev + L_next) / 2.0

        # 指标的当前项: (S_i^2) * kappa_i
        # 这正比于 (v * delta_t)^2 * kappa = (v^2 * kappa) * (delta_t)^2
        # = (横向加速度) * (delta_t)^2
        metric_term = (S_i**2) * kappa_i
        total_metric += metric_term
    
    return total_metric
