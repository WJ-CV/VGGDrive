import numpy as np

from enum import Enum
from typing import TypedDict


import json

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


class CateResultDict(TypedDict):
    category_str: str
    classifier_type: str
    fail_msg: str
    subtype_index: int

class EgoLonActionCategory(Enum):
    MetaAction_Stop = 1
    MetaAction_Start = 2
    MetaAction_HardBrake = 3
    MetaAction_Decelerate = 4
    MetaAction_Maintain = 5
    MetaAction_Accelerate = 6
    MetaAction_HardAccelerate = 7
    MetaAction_Unknown = 8

class EgoLongitudinalMetaActionClassifier:
    def __init__(self):
        self.hard_brake_th = -3.0
        self.decelerate_th = -1.0
        self.accelerate_th = 1.0
        self.hard_accelerate_th = 3.0
        self.parking_distance_th = 1.0
        self.start_speed_th = 0.2
        self.acc_calc_index = ((0, 2), (2, 4))
        self.segments = ((0, 3), (3, 6))
        self.dt = 0.1

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _check_start(self, trajectory):
        end_speed = trajectory[-1, 2]
        start_flag = (trajectory[:3, 2] <= self.start_speed_th).all()
        if start_flag and end_speed >= 2.0:
            return True, EgoLonActionCategory.MetaAction_Start
        return False, EgoLonActionCategory.MetaAction_Unknown

    def _check_stop(self, trajectory):
        dist = np.linalg.norm(trajectory[-1, :2] - trajectory[0, :2])
        
        if dist < self.parking_distance_th and len(trajectory) > 5:
            return True, EgoLonActionCategory.MetaAction_Stop
        return False, EgoLonActionCategory.MetaAction_Unknown

    def _check_acc(self, trajectory):
        speed = trajectory[:, -1]
        first_seg_index = (self.acc_calc_index[0][0], min(len(speed), self.acc_calc_index[0][1]))
        first_seg_speed = speed[first_seg_index[0] : first_seg_index[1]].mean()
        if self.acc_calc_index[1][0] >= len(speed):
            return False, EgoLonActionCategory.MetaAction_Unknown
        
        last_seg_index = (self.acc_calc_index[1][0], min(len(speed), self.acc_calc_index[1][1]))
        last_seg_speed = speed[last_seg_index[0] : last_seg_index[1]].mean()
        
        time_interval = (last_seg_index[1] + last_seg_index[0] - first_seg_index[1] - first_seg_index[0]) * 0.5 * self.dt
        acc = (last_seg_speed - first_seg_speed) / time_interval if time_interval != 0 else 0
        
        if acc < self.hard_brake_th:
            return True, EgoLonActionCategory.MetaAction_HardBrake
        elif self.hard_brake_th <= acc < self.decelerate_th:
            return True, EgoLonActionCategory.MetaAction_Decelerate
        elif self.decelerate_th <= acc < self.accelerate_th:
            return True, EgoLonActionCategory.MetaAction_Maintain
        elif self.accelerate_th <= acc < self.hard_accelerate_th:
            return True, EgoLonActionCategory.MetaAction_Accelerate
        else:
            return True, EgoLonActionCategory.MetaAction_HardAccelerate

    def process(self, meta_data) -> CateResultDict:
        if not hasattr(meta_data, 'ego_fut_info'):
            return self._error_result("Invalid meta_data")
            
        fut_info = meta_data.ego_fut_info
        valid_mask = fut_info.masks != 0
        trajs = fut_info.trajs[valid_mask]
        velocities = fut_info.velocity[valid_mask]
        
        if len(trajs) < 2:
            return self._error_result("Insufficient trajectory points")
        trajectory = np.concatenate([trajs, velocities], axis=-1)
        behaviors = []
        
        for seg in self.segments:
            if seg[0] >= len(trajectory):
                break
                
            seg_traj = trajectory[seg[0]:seg[1]]
            for check_func in [self._check_start, self._check_stop, self._check_acc]:
                flag, cate = check_func(seg_traj)
                if flag:
                    behaviors.append(cate)
                    break
            else:
                behaviors.append(EgoLonActionCategory.MetaAction_Unknown)
                
        return CateResultDict(
            category_str=behaviors[0].name if behaviors else EgoLonActionCategory.MetaAction_Unknown.name,
            classifier_type=self.name,
            fail_msg="" if behaviors else "No behaviors detected",
            subtype_index=0
        )

    def _error_result(self, msg: str) -> CateResultDict:
        return CateResultDict(
            category_str=EgoLonActionCategory.MetaAction_Unknown.name,
            classifier_type=self.name,
            fail_msg=msg,
            subtype_index=0
        )

# 直行/左转/右转/左侧偏移/右侧偏移/掉头
class EgoLatActionCategory(Enum):
    MetaAction_GoStraight = 1
    MetaAction_TurnLeft = 2
    MetaAction_TurnRight = 3
    MetaAction_DeviateLeft = 4
    MetaAction_DeviateRight = 5
    MetaAction_UTurn = 6
    MetaAction_Unknown = 7

class EgoFutInfo:
    """完全保持用户原始数据结构"""
    def __init__(self, trajs, velocity, masks):
        self.trajs = trajs    # 轨迹坐标 (N,2)
        self.velocity = velocity  # 速度数组 (N,2)
        self.masks = masks    # 有效点掩码 (N,)

class FrameInfo:
    """完全保持用户原始数据结构"""
    def __init__(self, ego_fut_info=None):
        self.ego_fut_info = ego_fut_info

class CateResultDict(dict):
    def __init__(self, category_str: str, classifier_type: str, fail_msg: str = "", subtype_index: int = 0):
        super().__init__(
            category_str=category_str,
            classifier_type=classifier_type,
            fail_msg=fail_msg,
            subtype_index=subtype_index
        )
        
EPSILON = 1e-2


def compute_angle(v1, v2):
    if np.linalg.norm(v1) < 1e-5 or np.linalg.norm(v2) < 1e-5:
        return 0
    else:
        arccos = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return abs(np.degrees(arccos))


def assign_segment_behavior(
    segment, turn_angle_th=40, uturn_angle_th=120, displacement_threshold=1.0
):
    """
    判断单段轨迹的行为
    :param segment: 单段轨迹，格式为 [(x1, y1, heading1), (x2, y2, heading2), ...]
    :param turn_angle_th: 转弯角度阈值（单位：度）
    :param uturn_angle_th: 掉头角度阈值 (单位：度)
    :param displacement_threshold: 横向位移变化的阈值（单位：米）
    :return: 单段轨迹的行为类别
    """
    initial_x, initial_y, initial_heading = segment[0]
    final_x, final_y, final_heading = segment[-1]

    """
        y (longitude)
        ^
        |
        |
        |
    (0.,0.) -------> x (lateral)
    
    """
    # 计算方向变化
    heading_change = final_heading - initial_heading
    heading_change = (heading_change + 180) % 360 - 180  # 归一化到 [-180, 180]

    # 计算横向位移变化
    displacement = final_x - initial_x

    # 判断行为类别
    if abs(heading_change) > uturn_angle_th:  # 掉头
        return EgoLatActionCategory.MetaAction_UTurn
    elif heading_change > turn_angle_th:  # 左转
        return EgoLatActionCategory.MetaAction_TurnLeft
    elif heading_change < -turn_angle_th:  # 右转
        return EgoLatActionCategory.MetaAction_TurnRight
    elif displacement > displacement_threshold and heading_change < -5:  # 右侧偏移
        return EgoLatActionCategory.MetaAction_DeviateRight
    elif displacement < -displacement_threshold and heading_change > 5:  # 左侧偏移
        return EgoLatActionCategory.MetaAction_DeviateLeft
    else:  # 直行
        return EgoLatActionCategory.MetaAction_GoStraight


def assign_lateral_behavior(
    trajectory,
    turn_angle_th=40,
    uturn_angle_th=120,
    displacement_threshold=1.0,
    min_segment_length=5,
    max_segment_length=20,
    curvature_threshold=0.1,
):
    """
    判断自车的横向行为，支持动态分段分析
    :param trajectory: 自车未来轨迹，格式为 [(x1, y1, heading1), (x2, y2, heading2), ...]
    :param turn_angle_th: 转弯角度阈值（单位：度）
    :param uturn_angle_th: 掉头角度阈值 (单位：度)
    :param displacement_threshold: 横向位移变化的阈值（单位：米）
    :param min_segment_length: 最小分段长度（时间步数）
    :param max_segment_length: 最大分段长度（时间步数）
    :param curvature_threshold: 曲率阈值，用于动态调整分段长度
    :return: 横向行为类别列表
    """
    behaviors = []
    n = len(trajectory)
    i = 0

    while i < n:
        # 动态确定分段长度
        segment_length = min_segment_length
        if i + max_segment_length < n:
            # 计算当前段的曲率
            segment = trajectory[i : i + max_segment_length]
            headings = [point[2] for point in segment]
            heading_changes = np.abs(np.diff(headings))
            avg_heading_change = np.mean(heading_changes)

            # 根据曲率调整分段长度
            if avg_heading_change > curvature_threshold:
                segment_length = min_segment_length  # 曲率大，分段长度缩短
            else:
                segment_length = max_segment_length  # 曲率小，分段长度延长

        # 截取当前段
        segment = trajectory[i : i + segment_length]
        if len(segment) < 2:
            break
        # 判断当前段的行为
        behavior = assign_segment_behavior(
            segment,
            turn_angle_th=turn_angle_th,
            uturn_angle_th=uturn_angle_th,
            displacement_threshold=displacement_threshold,
        )
        behaviors.append(behavior)

        # 移动到下一段
        i += segment_length

    return behaviors


class EgoLateralMetaActionClassifier:
    def __init__(self):
        self.turn_angle_th = 20
        self.uturn_angle_th = 120
        self.displacement_threshold = 1.5
        self.min_segment_length = 30  # 最小以3s作为判断
        self.max_segment_length = 60  # 最大以6s作为判断
        self.curvature_threshold = 1  # 平均0.1s内变化超过度数

    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    def _check_uturn(self, ego_fut_yaw):
        heading_change = ego_fut_yaw - ego_fut_yaw[0]
        heading_change = (heading_change + 180) % 360 - 180
        if (abs(heading_change) > self.uturn_angle_th).any():
            return True, EgoLatActionCategory.MetaAction_UTurn
        else:
            return False, EgoLatActionCategory.MetaAction_Unknown

    def process(self, meta_data: FrameInfo) -> CateResultDict:
        
        # get ego future trajectory
        ego_fut_mask = meta_data.ego_fut_info.masks
        ego_fut_traj = meta_data.ego_fut_info.trajs
        ego_fut_traj = ego_fut_traj[ego_fut_mask != 0]
        if len(ego_fut_traj) <= 1:
            return CateResultDict(
                category_str=EgoLatActionCategory.MetaAction_Unknown.name,
                classifier_type=self.name,
                fail_msg="Empty ego fut traj or no valid points",
            )

        ego_fut_yaw = np.full(
            len(ego_fut_traj), np.pi / 2
        )  # 默认值为朝向自车坐标系正前方
        deltas = np.diff(ego_fut_traj, axis=0)
        delta_norms = np.linalg.norm(deltas, axis=1)

        # 计算航向角
        for i in range(len(deltas)):
            if delta_norms[i] < EPSILON:
                # 差值向量接近于 (0, 0)，使用前一个有效航向角
                if i > 0:
                    ego_fut_yaw[i] = ego_fut_yaw[i - 1]
            else:
                # 计算航向角
                ego_fut_yaw[i] = np.arctan2(deltas[i, 1], deltas[i, 0])

        # 处理最后一个点的航向角
        ego_fut_yaw[-1] = ego_fut_yaw[-2]
        ego_fut_yaw = ego_fut_yaw * 180.0 / np.pi

        # special check for uturn
        flag, cate = self._check_uturn(ego_fut_yaw)
        if flag:
            return CateResultDict(
                category_str=cate.name, classifier_type=self.name, fail_msg=""
            )

        behaviors = assign_lateral_behavior(
            np.concatenate([ego_fut_traj, ego_fut_yaw[:, np.newaxis]], axis=-1),
            turn_angle_th=self.turn_angle_th,
            uturn_angle_th=self.uturn_angle_th,
            displacement_threshold=self.displacement_threshold,
            min_segment_length=self.min_segment_length,
            max_segment_length=self.max_segment_length,
            curvature_threshold=self.curvature_threshold,
        )

        if not behaviors:
            return CateResultDict(
                category_str=EgoLatActionCategory.MetaAction_Unknown.name,
                classifier_type=self.name,
                fail_msg="",
            )
        else:
            # 目前使用第一段的行为作为最终标签
            return CateResultDict(
                category_str=behaviors[0].name, classifier_type=self.name, fail_msg=""
            )

def safe_round(number, ndigits=1):
    rounded = round(number, ndigits)  # Round to 3 decimal places
    return rounded + 0.0  # Convert -0.0 to 0.0

def preprocess_meta_action_str(s):
    action_trans_dict = {
        'Stop': 'Slow Down to Stop',
        'Start': 'Start from Zero',
        'HardBrake': 'Hard Brake',
        'Decelerate': 'Decelerate',
        'Maintain': 'Maintain Speed',
        'Accelerate': 'Accelerate',
        'HardAccelerate': 'Hard Accelerate',
        'GoStraight': 'Go Straight',
        'TurnLeft': 'Turn Left',
        'TurnRight': 'Turn Right',
        'DeviateLeft': 'Deviate Left',
        'DeviateRight': 'Deviate Right',
        'UTurn': 'U Turn',
        'Unknown': 'Unknown'
    }
    return action_trans_dict[s]

json_query_prompt = '''# Role
An experienced driver utilizing a vehicle equipped with advanced vision systems. 

# Profile
- Familiar with traffic laws and road safety.
- Skilled in analyzing road conditions, behaviors of surrounding vehicles, and other relevant environmental factors.
- Proficient in interpreting continuous streams of front-view images from a first-person perspective. 

# Goals
To predict the most likely lateral control, longitudinal control, and future trajectory of the vehicle within the ego coordinate system using the provided visual data.

# Requirements for Output
- **lateral control**: 
  - Indicates vehicle directional action along the lateral axis.
  - Options: ["Maintain Speed", "Accelerate", "Decelerate", "Start from Zero", "Slow Down to Stop", "Hard Accelerate", "Hard Brake", "Unknown"].

- **longitudinal control**: 
  - Represents vehicle control action regarding speed along the longitudinal axis.
  - Options: ["Go Straight", "Turn Left", "Turn Right", "Deviate Left", "Deviate Right", "U Turn", "Unknown"].

- **trajectory**: 
  - A list of tuples outlining the predicted path of the vehicle.
  - Each tuple includes two float values representing the x and y coordinates of points along the trajectory.
  - Predict the trajectory over **six** steps at intervals of 1/2 second.

# Output Format

```json
{
    "lateral_control": "<lateral_control>",
    "longitudinal_control": "<longitudinal_control>",
    "trajectory": [
        [<x_coord>, <y_coord>],
        [<x_coord>, <y_coord>],
        [<x_coord>, <y_coord>],
        [<x_coord>, <y_coord>],
        [<x_coord>, <y_coord>],
        [<x_coord>, <y_coord>]
    ]
}
```'''

angle_query_prompt = '''# Role
An experienced driver utilizing a vehicle equipped with advanced vision systems. 

# Profile
- Familiar with traffic laws and road safety.
- Skilled in analyzing road conditions, behaviors of surrounding vehicles, and other relevant environmental factors.
- Proficient in interpreting continuous streams of front-view images from a first-person perspective. 

# Goals
To predict the most likely lateral control, longitudinal control, and future trajectory of the vehicle within the ego coordinate system using the provided visual data.

# Requirements for Output
- **lateral control**: 
  - Indicates vehicle directional action along the lateral axis.
  - Options: ["Maintain Speed", "Accelerate", "Decelerate", "Start from Zero", "Slow Down to Stop", "Hard Accelerate", "Hard Brake", "Unknown"].

- **longitudinal control**: 
  - Represents vehicle control action regarding speed along the longitudinal axis.
  - Options: ["Go Straight", "Turn Left", "Turn Right", "Deviate Left", "Deviate Right", "U Turn", "Unknown"].

- **trajectory**: 
  - A list of tuples outlining the predicted path of the vehicle.
  - Each tuple includes two float values representing the x and y coordinates of points along the trajectory.
  - Predict the trajectory over **six** steps at intervals of 1/2 second.

# Output Format

<lateral_control> your choice here </lateral_control>
<longitudinal_control> your choice here </longitudinal_control>
<trajectory> [[<x_coord_1>, <y_coord_1>], [<x_coord_2>, <y_coord_2>], [<x_coord_3>, <y_coord_3>], [<x_coord_4>, <y_coord_4>], [<x_coord_5>, <y_coord_5>], [<x_coord_6>, <y_coord_6>]] </trajectory>'''


json_speed_query_prompt = '''# Role
An experienced driver utilizing a vehicle equipped with advanced vision systems. 

# Profile
- Familiar with traffic laws and road safety.
- Skilled in analyzing road conditions, behaviors of surrounding vehicles, and other relevant environmental factors.
- Proficient in interpreting continuous streams of front-view images from a first-person perspective. 
- The current speed is __speed__ m/s.

# Goals
To predict the most likely lateral control, longitudinal control, and future trajectory of the vehicle within the ego coordinate system using the provided visual data.

# Requirements for Output
- **lateral control**: 
  - Indicates vehicle directional action along the lateral axis.
  - Options: ["Go Straight", "Turn Left", "Turn Right", "Deviate Left", "Deviate Right", "U Turn", "Unknown"].

- **longitudinal control**: 
  - Represents vehicle control action regarding speed along the longitudinal axis.
  - Options: ["Maintain Speed", "Accelerate", "Decelerate", "Start from Zero", "Slow Down to Stop", "Hard Accelerate", "Hard Brake", "Unknown"].

- **trajectory**: 
  - A list of tuples outlining the predicted path of the vehicle.
  - Each tuple includes two float values representing the x and y coordinates of points along the trajectory.
  - Predict the trajectory over **six** steps at intervals of 1/2 second.

# Output Format

```json
{
    "lateral_control": "<lateral_control>",
    "longitudinal_control": "<longitudinal_control>",
    "trajectory": [
        [<x_coord>, <y_coord>],
        [<x_coord>, <y_coord>],
        [<x_coord>, <y_coord>],
        [<x_coord>, <y_coord>],
        [<x_coord>, <y_coord>],
        [<x_coord>, <y_coord>]
    ]
}
```'''

angle_speed_query_prompt = '''# Role
An experienced driver utilizing a vehicle equipped with advanced vision systems. 

# Profile
- Familiar with traffic laws and road safety.
- Skilled in analyzing road conditions, behaviors of surrounding vehicles, and other relevant environmental factors.
- Proficient in interpreting continuous streams of front-view images from a first-person perspective. 
- The current speed is __speed__ m/s.

# Goals
To predict the most likely lateral control, longitudinal control, and future trajectory of the vehicle within the ego coordinate system using the provided visual data.

# Requirements for Output
- **lateral control**: 
  - Indicates vehicle directional action along the lateral axis.
  - Options: ["Go Straight", "Turn Left", "Turn Right", "Deviate Left", "Deviate Right", "U Turn", "Unknown"].

- **longitudinal control**: 
  - Represents vehicle control action regarding speed along the longitudinal axis.
  - Options: ["Maintain Speed", "Accelerate", "Decelerate", "Start from Zero", "Slow Down to Stop", "Hard Accelerate", "Hard Brake", "Unknown"].

- **trajectory**: 
  - A list of tuples outlining the predicted path of the vehicle.
  - Each tuple includes two float values representing the x and y coordinates of points along the trajectory.
  - Predict the trajectory over **six** steps at intervals of 1/2 second.

# Output Format

<lateral_control> your choice here </lateral_control>
<longitudinal_control> your choice here </longitudinal_control>
<trajectory> [[<x_coord_1>, <y_coord_1>], [<x_coord_2>, <y_coord_2>], [<x_coord_3>, <y_coord_3>], [<x_coord_4>, <y_coord_4>], [<x_coord_5>, <y_coord_5>], [<x_coord_6>, <y_coord_6>]] </trajectory>'''



# query_thinking_prompt = '''# Role
# An experienced driver utilizing a vehicle equipped with advanced vision systems. 

# # Profile
# - Familiar with traffic laws and road safety.
# - Skilled in analyzing road conditions, behaviors of surrounding vehicles, and other relevant environmental factors.
# - Proficient in interpreting continuous streams of front-view images from a first-person perspective. 

# # Goals
# To predict the most likely lateral control, longitudinal control, and future trajectory of the vehicle within the ego coordinate system using the provided visual data.

# # Requirements for Output
# The analysis and prediction should be expressed in <think> reasoning process here </think> and <answer> answer here (with a structured JSON format) </answer>:

# Think carefully about all aspects first and place the reasoning process within <think> </think>.

# - **lateral control**: 
#   - Indicates vehicle directional action along the lateral axis.
#   - Options: ["Maintain Speed", "Accelerate", "Decelerate", "Start from Zero", "Slow Down to Stop", "Hard Accelerate", "Hard Brake", "Unknown"].

# - **longitudinal control**: 
#   - Represents vehicle control action regarding speed along the longitudinal axis.
#   - Options: ["Go Straight", "Turn Left", "Turn Right", "Deviate Left", "Deviate Right", "U Turn", "Unknown"].

# - **trajectory**: 
#   - A list of tuples outlining the predicted path of the vehicle.
#   - Each tuple includes two float values representing the x and y coordinates of points along the trajectory.

# # Time Step Requirements
# - Predict the trajectory over six steps.
# - Each step is at intervals of 1/2 second.

# # Constraints and Considerations
# - Road conditions, surrounding vehicle behavior, and environmental factors must be analyzed and factored into predictions.
# - Lateral control and longitudinal control must be chosen from the given set of options, and the trajectory must predict six steps.

# # Output Format

# <think> reasoning process here </think>
# <answer>
# ```json
# {
#     "lateral_control": "<lateral_control>",
#     "longitudinal_control": "<longitudinal_control>",
#     "trajectory": [
#         [<x_coord>, <y_coord>],
#         [<x_coord>, <y_coord>],
#         [<x_coord>, <y_coord>],
#         [<x_coord>, <y_coord>],
#         [<x_coord>, <y_coord>],
#         [<x_coord>, <y_coord>]
#     ]
# }
# ```
# </answer>'''

# angle_speed_r1_query_prompt = '''# Role
# Expert driver specializing in traffic laws and road safety, proficient in real-time analysis of road conditions, surrounding vehicle behavior, and environmental factors using first-person vision systems.

# # Goals
# Think, and then predict the vehicle's lateral/longitudinal control and trajectory in ego coordinates.

# ## Output Requirements

# ### **Reasoning process**
# - Identify and describe all relevant elements in the traffic scene, including vehicles, vulnerable road users (pedestrians, cyclists, etc.), traffic signs, signals, obstacles, weather, and road conditions (e.g., wet, muddy, blocked).
# - Provide driving advice based on real-time traffic conditions, complying with traffic laws and safety requirements.
# - Select appropriate driving control from the predefined set to ensure safe, efficient, and smooth navigation.

# ### **Controls**
# - **lateral control**: 
#   - Indicates vehicle directional action along the lateral axis.
#   - Options: ["Maintain Speed", "Accelerate", "Decelerate", "Start from Zero", "Slow Down to Stop", "Hard Accelerate", "Hard Brake", "Unknown"].

# - **longitudinal control**: 
#   - Represents vehicle control action regarding speed along the longitudinal axis.
#   - Options: ["Go Straight", "Turn Left", "Turn Right", "Deviate Left", "Deviate Right", "U Turn", "Unknown"].

# ### **Trajectory**
# - The current speed is __speed__ meters per second.
# - Six 0.5-second steps (3 seconds total) in ego coordinates **(x, y)** (unit: meters):
#   - **x (lateral)**:
#     - **> 0**: Right | **< 0**: Left | **= 0**: No lateral movement.
#   - **y (longitudinal)**:
#     - **> 0**: Forward | **< 0**: Backward | **= 0**: Stopped.

# ## Output Format
# <think> reasoning process here </think>
# <lateral_control> your choice here </lateral_control>
# <longitudinal_control> your choice here </longitudinal_control>
# <trajectory> [[<x1>, <y1>], [<x2>, <y2>], [<x3>, <y3>], [<x4>, <y4>], [<x5>, <y5>], [<x6>, <y6>]] </trajectory>'''

# angle_speed_r1_query_prompt = '''You are an expert driver who complies with **traffic laws and road safety**. These images are from a **first-person perspective (ego view)**, with the ego vehicle positioned at the **bottom center**. Your task is to predict the ego vehicle's behavior over the next 3 seconds, including:

# Lateral control (choose one from: [Go Straight, Turn Left, Turn Right, Deviate Left, Deviate Right, U-Turn]).

# Longitudinal control (choose one from: [Maintain Speed, Accelerate, Decelerate, Start from Zero, Slow Down to Stop, Hard Accelerate, Hard Brake]).

# Trajectory, represented by six (x, y) coordinates (unit: meters), where x > 0 is rightward movement, x < 0 is leftward movement, y > 0 is forward movement, and y < 0 is backward movement. The current speed is __speed__ m/s.

# Output format:
# <think>reasoning process here</think>
# <lateral_control>your choice here</lateral_control>
# <longitudinal_control>your choice here</longitudinal_control>
# <trajectory>[[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6]]</trajectory>'''

angle_speed_r1_cold_start_query_prompt = '''You are an expert driver who complies with **traffic laws and road safety**. These images are from a **first-person perspective (ego view)**, with the ego vehicle positioned at the **bottom center**. Your task is to predict the ego vehicle's behavior over the next 3 seconds, including:

Lateral control (choose one from: [Go Straight, Turn Left, Turn Right, Deviate Left, Deviate Right, U-Turn]).

Longitudinal control (choose one from: [Maintain Speed, Accelerate, Decelerate, Start from Zero, Slow Down to Stop, Hard Accelerate, Hard Brake]).

Trajectory, represented by six (x, y) coordinates (unit: m, Ego Coordinates, 0.5s Intervals, 3s Total). The current speed is __speed__ m/s, where x > 0 is rightward, x < 0 is leftward, y > 0 is forward, and y < 0 is backward. The range of x is 0 to 15, and the range of y is 0 to 50.

Output format:
<lateral_control>your choice here</lateral_control>
<longitudinal_control>your choice here</longitudinal_control>
<trajectory>[[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6]]</trajectory>'''

angle_speed_r1_query_prompt = '''You are an expert driver who complies with **traffic laws and road safety**. These images are from a **first-person perspective (ego view)**, with the ego vehicle positioned at the **bottom center**. Your task is to predict the ego vehicle's behavior over the next 3 seconds, including:

Lateral control (choose one from: [Go Straight, Turn Left, Turn Right, Deviate Left, Deviate Right, U-Turn]).

Longitudinal control (choose one from: [Maintain Speed, Accelerate, Decelerate, Start from Zero, Slow Down to Stop, Hard Accelerate, Hard Brake]).

Trajectory, represented by six (x, y) coordinates (unit: m, Ego Coordinates, 0.5s Intervals, 3s Total). The current speed is __speed__ m/s, where x > 0 is rightward, x < 0 is leftward, y > 0 is forward, and y < 0 is backward. The range of x is 0 to 15, and the range of y is 0 to 50.

Output format:
<think>reasoning process here</think>
<lateral_control>your choice here</lateral_control>
<longitudinal_control>your choice here</longitudinal_control>
<trajectory>[[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6]]</trajectory>'''


angle_speed_r1_sft_query_prompt = '''You are an expert driver who complies with **traffic laws and road safety**. These images are from a **first-person perspective (ego view)**, with the ego vehicle positioned at the **bottom center**. Your task is to predict the ego vehicle's behavior over the next 3 seconds, including:

Lateral control (choose one from: [Go Straight, Turn Left, Turn Right, Deviate Left, Deviate Right, U-Turn]).

Longitudinal control (choose one from: [Maintain Speed, Accelerate, Decelerate, Start from Zero, Slow Down to Stop, Hard Accelerate, Hard Brake]).

Trajectory, represented by six (x, y) coordinates (unit: m, Ego Coordinates, 0.5s Intervals, 3s Total). The current speed is __speed__ m/s, where x > 0 is rightward, x < 0 is leftward, y > 0 is forward, and y < 0 is backward. The range of x is 0 to 15, and the range of y is 0 to 50.

Output format:
<lateral_control>your choice here</lateral_control>
<longitudinal_control>your choice here</longitudinal_control>
<trajectory>[[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6]]</trajectory>'''


angle_speed_r1_sft_around_query_prompt = '''You are an expert driver who complies with **traffic laws and road safety**. Your task is to predict the ego vehicle's behavior over the next 3 seconds, including:

Lateral control (choose one from: [Go Straight, Turn Left, Turn Right, Deviate Left, Deviate Right, U-Turn]).

Longitudinal control (choose one from: [Maintain Speed, Accelerate, Decelerate, Start from Zero, Slow Down to Stop, Hard Accelerate, Hard Brake]).

Trajectory, represented by six (x, y) coordinates (unit: m, Ego Coordinates, 0.5s Intervals, 3s Total). The current speed is __speed__ m/s, where x > 0 is rightward, x < 0 is leftward, y > 0 is forward, and y < 0 is backward. The range of x is 0 to 15, and the range of y is 0 to 50.

Output format:
<lateral_control>your choice here</lateral_control>
<longitudinal_control>your choice here</longitudinal_control>
<trajectory>[[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6]]</trajectory>'''


angle_m_speed_r1_query_prompt = '''You are an expert driver who complies with **traffic laws and road safety**. These images are from a **first-person perspective (ego view)**, with the ego vehicle positioned at the **bottom center**.

### **Step 1: Thinking Process**
1. **Driving Environment Perception**:
   - Analyze the road layout (e.g., straight road, intersection, curves).
   - Detect surrounding objects (vehicles, pedestrians, obstacles).
   - Observe traffic signals (traffic lights, signs, lane markings).
   - Assess weather and lighting conditions (day/night, rain, fog).

2. **Prediction & Planning**:
   - Estimate the behavior of other road users.
   - Predict potential hazards (sudden braking, pedestrians crossing).
   - Determine safe speed and trajectory based on road conditions.
   - Ensure compliance with traffic laws and safety margins.

### **Step 2: Predicted Meta Actions & Trajectory (Next 3 Seconds)**
1. **Lateral Control (Steering/Direction)**:
   - Choose one from: [Go Straight, Turn Left, Turn Right, Deviate Left, Deviate Right, U-Turn]

2. **Longitudinal Control (Speed)**:
   - Choose one from: [Maintain Speed, Accelerate, Decelerate, Start from Zero, Slow Down to Stop, Hard Accelerate, Hard Brake]

3. **Trajectory (Ego Coordinates, 0.5s Intervals, 3s Total)**:
   - Current speed: **__speed__ m/s**
   - Predicted coordinates (unit: meters):
     - **x (lateral)**: `> 0` = Right; `< 0` = Left; `= 0` = Straight.
     - **y (longitudinal)**: `> 0` = Forward; `< 0` = Backward; `= 0` = Stopped.

### Output Format:
<think>reasoning process here</think>
<lateral_control>your choice here</lateral_control>
<longitudinal_control>your choice here</longitudinal_control>
<trajectory>[[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6]]</trajectory>'''

QUERY_PROMPT = {
    'json': json_query_prompt,
    'angle': angle_query_prompt,
    'json_speed': json_speed_query_prompt,
    'angle_speed': angle_speed_query_prompt,
    'angle_speed_r1': angle_speed_r1_query_prompt,
    'angle_m_speed_r1': angle_m_speed_r1_query_prompt,
    'angle_speed_r1_cold_start': angle_speed_r1_cold_start_query_prompt,
    'angle_speed_r1_sft': angle_speed_r1_sft_query_prompt,
    'angle_speed_r1_sft_around': angle_speed_r1_sft_around_query_prompt
}
