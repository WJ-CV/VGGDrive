import os
import json
import cv2
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import random

from collections import defaultdict

import pickle

from open_r1.dataloader.tools.traj2angle import calculate_steering_angle
from open_r1.dataloader.tools.speed2acc import calculate_acceleration
from open_r1.dataloader.tools.traj_bias import compute_trajectory_bias
from open_r1.dataloader.tools.traj2ka import calculate_trajectory_dynamic_strain_index
from open_r1.dataloader.tools.sampler import sample_frames, stratified_sampling
from open_r1.dataloader.tools.nav_info import get_nav_info

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle_chunks(prefix, data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        file_path = prefix.replace('.pkl', f"_{i//chunk_size}.pkl")
        save_pickle(file_path, chunk)

def load_pickle_chunks(prefix):
    data = []
    i = 0
    while True:
        file_path = prefix.replace('.pkl', f"_{i}.pkl")
        if not os.path.exists(file_path):
            break
        print(f'Loading {file_path} ..')
        data.extend(load_pickle(file_path))
        i += 1
    return data


def mean(l):
    if not isinstance(l, list):
        return l
    if len(l) == 0:
        return 0
    return sum(l) / len(l)

from open_r1.dataloader.utils import safe_round

import cv2

class CoVLADataset:
    def __init__(
        self,
        data_path: str,
        data_args
    ):
        """
        Initialize the CoVLA dataset loader.
        
        Args:
            data_path: Path to the video dataset
            states_path: Path to the states directory containing JSONL files
        """
        self.meta_file = os.path.join(data_args.meta_dir, f"covla_cache_scenes_8_5k.pkl") if data_args.split == 'train' else os.path.join(data_args.meta_dir, f"covla_cache_scenes_1_5k.pkl")
        self.meta_scene_file = os.path.join(data_args.meta_dir, f"covla_cache_scenes_8_5k.json") if data_args.split == 'train' else os.path.join(data_args.meta_dir, f"covla_cache_scenes_1_5k.json")
        self.data_path = data_path
        self.data_args = data_args
        self.obs_len = 4

        self.videos_path = os.path.join(data_path, 'videos')
        self.states_path = os.path.join(data_path, 'states')
        self.frames_path = os.path.join(data_path, 'frames')
        self._scenes = []
        self._samples = []

        self.decimal = data_args.decimal

        self.resize_kwargs = {
            "resized_height": 448,
            "resized_width": 896
        }

        self.configs = {
            '__steering_threshold_1__': 3,
            '__steering_threshold_2__': 20,
            '__steering_threshold_3__': 60,
            '__throttle_threshold_1__': 3,
            '__throttle_threshold_2__': 20,
            '__throttle_threshold_3__': 35,
            '__slope_threshold_1__': 0.05,
        }

        self.prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", f'{data_args.query_prompt_type}.txt')

        if data_args.force_load or len(data_args.cache_file) > 0 and os.path.isfile(data_args.cache_file):
            print(f'检测到cache file，从 {data_args.cache_file} 中加载 ..')
            self._samples = load_json(data_args.cache_file)
        else:
            # Initialize the dataset
            self._prepare_scenes()

            self._load_training_samples()
            if len(data_args.cache_file) > 0:
                print(f'保存到cache file {data_args.cache_file} ..')
                save_json(data_args.cache_file, self._samples)

        if data_args.sample_size > 0:
            assert len(data_args.sample_ratios.split(',')) == len(data_args.sample_bins.split(',')) - 1, f"{len(data_args.sample_ratios.split(','))}, {len(data_args.sample_bins.split(','))}"
            self._samples = stratified_sampling(self._samples, [float(i) for i in data_args.sample_bins.split(',')], [int(i) for i in data_args.sample_ratios.split(',')], data_args.sample_size)
        self._samples = [self.process_item(idx) for idx in range(len(self._samples))]

    # 转向类别分类
    def classify_steering(self, angle):
        if angle < -self.configs['__steering_threshold_3__']:
            return 'Turn Right'
        elif -self.configs['__steering_threshold_3__'] <= angle < -self.configs['__steering_threshold_2__']:
            return 'Moderate Right'
        elif -self.configs['__steering_threshold_2__'] <= angle < -self.configs['__steering_threshold_1__']:
            return 'Slight Right'
        elif -self.configs['__steering_threshold_1__'] <= angle <= self.configs['__steering_threshold_1__']:
            return 'Go Straight'
        elif self.configs['__steering_threshold_1__'] < angle <= self.configs['__steering_threshold_2__']:
            return 'Slight Left'
        elif self.configs['__steering_threshold_2__'] < angle <= self.configs['__steering_threshold_3__']:
            return 'Moderate Left'
        else:
            return 'Turn Left'

    # 纵向类别分类
    def classify_throttle(self, throttle, brake):
        throttle = throttle * 100
        if brake:
            return 'Braking'
        if throttle <= self.configs['__throttle_threshold_1__']:
            return 'Coasting'
        elif throttle <= self.configs['__throttle_threshold_2__'] and throttle > self.configs['__throttle_threshold_1__']:
            return 'Light Accelerate'
        elif throttle <= self.configs['__throttle_threshold_3__'] and throttle > self.configs['__throttle_threshold_2__']:
            return 'Middle Accelerate'
        elif throttle > self.configs['__throttle_threshold_3__']:
            return 'Hard Accelerate'
        else:
            raise NotImplementedError

    def classify_slope(self, slope_interval):
        if slope_interval <= self.configs['__slope_threshold_1__'] and slope_interval > -self.configs['__slope_threshold_1__']:
            return 'Level ground'
        elif slope_interval <= -self.configs['__slope_threshold_1__']:
            return 'Downhill'
        elif slope_interval > self.configs['__slope_threshold_1__']:
            return 'Uphill'

    def format_frames(self, frames):
        return [{"type": "image", "image": f"file://{os.path.join(self.frames_path, i_frame)}", **self.resize_kwargs} for i_frame in frames]

    def format_single_sample(self, info):
        with open(self.prompt_file, 'r', encoding='utf-8') as file:
            prompt = file.read()

        temp_info = self.configs
        temp_info.update(info)
        temp_log = {}
        for placeholder, value in temp_info.items():
            if ('__' in placeholder or 'bef1s' in placeholder or 'aft1s' in placeholder) and '_threshold_' not in placeholder:
                temp_log[placeholder] = value
            if '__' in placeholder and placeholder in prompt:
                prompt = prompt.replace(placeholder, f'{value:.2f}' if not isinstance(value, str) and not isinstance(value, int) else f'{value}')

        traj_str = [f"[{safe_round(x, self.decimal)}, {safe_round(y, self.decimal)}]" for (x, y) in info['fut_trajectory']]
        traj_str = '[' + ', '.join(traj_str) + ']'
        temp_log['traj_str'] = traj_str
        # NOTE: get biased samples
        steering_bias, throttle_bias = False, False
        if info["bef1s_steering"] == info['aft1s_steering'] and info["bef1s_throttle"] == info['aft1s_throttle']:
            steering_bias, throttle_bias = True, True
        elif info["bef1s_steering"] == info['aft1s_steering'] and info["bef1s_throttle"] != info['aft1s_throttle']:
            steering_bias = True
        elif info["bef1s_steering"] != info['aft1s_steering'] and info["bef1s_throttle"] == info['aft1s_throttle']:
            throttle_bias = True

        # def compute_trajectory_bias(train_val_traj):
        #     import numpy as np
        #     from statsmodels.tsa.arima.model import ARIMA
        #     from sklearn.metrics import mean_squared_error

        #     def forecast_with_arma(data, steps=20, order=(1,0,1)):
        #         model = ARIMA(data, order=order)
        #         model_fit = model.fit()
        #         forecast = model_fit.forecast(steps=steps)
        #         return forecast

        #     x_forecast = forecast_with_arma([i[0] for i in train_val_traj[:40]])
        #     y_forecast = forecast_with_arma([i[1] for i in train_val_traj[:40]])
        #     extended_trajectory = [[x, y] for x, y in zip(x_forecast, y_forecast)]

        #     def calculate_metrics(predicted, gt):
        #         """Calculate various comparison metrics"""
                
        #         rmse_total = np.sqrt(mean_squared_error(gt, predicted))
                
        #         return rmse_total

        #     return calculate_metrics(np.array(extended_trajectory), np.array(train_val_traj)[40:, :2])

        # def compute_trajectory_bias(train_val_traj):
        #     extended_trajectory = extend_trajectory_kinematic(train_val_traj[:40])
        #     from sklearn.metrics import mean_squared_error

        #     def calculate_metrics(predicted, gt):
        #         """Calculate various comparison metrics"""
                
        #         rmse_total = np.sqrt(mean_squared_error(gt, predicted))
                
        #         return rmse_total

        #     return calculate_metrics(extended_trajectory, np.array(train_val_traj)[40:])

        if len(info['raw_all_trajectory']) < 60:
            return None
        try:
            trajectory_bias, extended_trajectory, extended_trajectory_gt = compute_trajectory_bias(info['raw_all_trajectory'], 40, 20, 3)
            return {
                # "messages": [
                #     {
                #         "role": "user",
                #         "content":
                #             self.format_frames(info['frames']) + \
                #             [{"type": "text", "text": prompt}],
                #     },
                #     {
                #         "role": "assistant",
                #         "content": f'<steering_control>{info["gt_steering_control"]}</steering_control>\n<throttle_and_brake_control>{info["gt_throttle_and_brake_control"]}</throttle_and_brake_control>\n<trajectory>{traj_str}</trajectory>',
                #     }
                # ],
                'id': info['id'],
                'frames': info['frames'],
                'steering_control': info["gt_steering_control"],
                'throttle_and_brake_control': info["gt_throttle_and_brake_control"],
                'trajectory': info['fut_trajectory'],
                'current_steering': info['__steering__'],
                'current_throttle': info['__throttle__'],
                'steering_bias': steering_bias,
                'throttle_bias': throttle_bias,
                'trajectory_bias': trajectory_bias,
                'trajectory_bias_pred': extended_trajectory,
                'trajectory_bias_gt': extended_trajectory_gt,
                **temp_log,
                'extrinsic_matrices': info['extrinsic_matrices'],
                'intrinsic_matrix': info['intrinsic_matrix']
            }
        except Exception as e:
            return None

    def _load_training_samples(self):
        def get_values(l, indices):
            return [l[i_l] for i_l in indices if i_l < len(l)]

        # debug_info = defaultdict(list)
        is_first = True

        for idx_scene, i_scene in enumerate(tqdm(self._scenes)):
            # if idx_scene > 60:
            #     break
            # 采样该场景下所有训练样本
            frames_indices = sample_frames(len(i_scene['frames']), frame_start_interval=self.data_args.frame_start_interval, ensure_no_overlap=self.data_args.ensure_no_overlap)

            if is_first:
                is_first = False
                print(f'Length of frames_indicessssssssssssssssssssssssssss: {len(frames_indices)}')
            # print(len(frames_indices))
            # for i_debug in frames_indices:
            #     print(i_debug)
            # assert 0

            for i_indices in frames_indices:
                current_frame_idx = i_indices[self.obs_len-1]
                info = {
                    'id': f"{i_scene['scene_token']}_" + "_".join([str(idx) for idx in i_indices]),
                    'scene_token': i_scene['scene_token'],
                    'frames': get_values(i_scene['frames'], i_indices[:self.obs_len]),
                    'fut_trajectory': get_values([i_traj[:2] for i_traj in i_scene['trajectory'][current_frame_idx]], [9, 19, 29, 39, 49, 59]),
                    'current_frame_idx': current_frame_idx,
                    'raw_all_trajectory': i_scene['trajectory'][i_indices[0]]
                }
                info.update({
                    k_scene: get_values(i_scene[k_scene], i_indices)
                    for k_scene in [
                        'speed',
                        'acceleration',
                        'gas',
                        'gas_pressed',
                        'steering_angle',
                        'brake',
                        'brake_pressed',
                        'gear',
                        'left_blinker',
                        'right_blinker',
                        'extrinsic_matrices',
                        'intrinsic_matrix',
                        'timestamps'
                    ]
                })

                slope_values = [i_traj[2] for i_traj in i_scene['trajectory'][max(0, info['current_frame_idx']-59)]]
                current_slope_gain = mean(slope_values[:20]) - mean(slope_values[-20:])

                bef1s_idx = max(info['current_frame_idx']-19, 0)
                cur_frame_end_idx = min(info['current_frame_idx']+1, len(i_scene['steering_angle']))
                def list_true_or_false(l):
                    return l.count(True) > len(l) / 2

                bef1s_traj = [i_traj[:2] for i_traj in i_scene['trajectory'][current_frame_idx-21]][:20]

                fut_steering_angle = mean(calculate_steering_angle(info['fut_trajectory']))
                info.update({
                    '__steering__': self.classify_steering(info['steering_angle'][self.obs_len]),
                    '__steering_degree__': mean(calculate_steering_angle(bef1s_traj)),
                    '__throttle__': self.classify_throttle(info['gas'][self.obs_len], info['brake_pressed'][self.obs_len]),
                    '__throttle_position__':  info["gas"][self.obs_len] * 100,
                    '__speed__': mean(i_scene['speed'][max(info['current_frame_idx']-19, 0): min(info['current_frame_idx']+1, len(i_scene['speed']))]),
                    # '__acceleration__': mean(i_scene['acceleration'][max(info['current_frame_idx']-9, 0): min(info['current_frame_idx']+1, len(i_scene['acceleration']))]),
                    '__acceleration__': mean(
                        calculate_acceleration(
                            i_scene['speed'][max(info['current_frame_idx']-19, 0): min(info['current_frame_idx']+1, len(i_scene['acceleration']))],
                        )
                    ),
                    '__slope__': self.classify_slope(current_slope_gain),
                    '__slope_gain__':  current_slope_gain,
                    'gt_steering_control': self.classify_steering(info['steering_angle'][self.obs_len + 1]),
                    'gt_throttle_and_brake_control': self.classify_throttle(info['gas'][self.obs_len + 1], info['brake_pressed'][self.obs_len + 1]),
                    'bef1s_steering': self.classify_steering(mean(i_scene['steering_angle'][bef1s_idx : cur_frame_end_idx])),
                    'bef1s_throttle': self.classify_throttle(mean(i_scene['gas'][bef1s_idx : cur_frame_end_idx]), list_true_or_false(i_scene['gas'][bef1s_idx : cur_frame_end_idx])),
                    'aft1s_steering': self.classify_steering(mean(i_scene['steering_angle'][info['current_frame_idx'] : info['current_frame_idx'] + 20])),
                    'aft1s_throttle': self.classify_throttle(mean(i_scene['gas'][info['current_frame_idx'] : info['current_frame_idx'] + 20]), list_true_or_false(i_scene['gas'][info['current_frame_idx'] : info['current_frame_idx'] + 20])),
                    'extrinsic_matrices': info["extrinsic_matrices"],
                    'intrinsic_matrix': info["intrinsic_matrix"],
                    'gold': {
                        'fut_steering_angle': fut_steering_angle,
                        'ka_fut': calculate_trajectory_dynamic_strain_index(info['fut_trajectory'])
                    }
                })
                info['__nav_info__'] = get_nav_info(fut_steering_angle, info['gold']['ka_fut'])

                current_sample = self.format_single_sample(info)

                if current_sample is not None:
                    self._samples.append(current_sample)

                # for k in info.keys():
                #     print(k)
                #     print(info[k])
                #     print()
                # assert 0
                # debug_info['steering_angle'].append(info['steering_angle'][self.obs_len])
                # debug_info['gas'].append(info['gas'][self.obs_len])
                # debug_info['brake_pressed'].append(info['brake_pressed'][self.obs_len])

        # save_json('debug_info.json', debug_info)

    def _prepare_scenes(self):
        training_scenes = load_json(self.meta_scene_file)

        if os.path.isfile(self.meta_file.replace('.pkl', '_0.pkl')):
            print(f"找到 {self.meta_file.replace('.pkl', '_0.pkl')}")
            self._scenes = load_pickle_chunks(self.meta_file)
            return
        print(f"未找到 {self.meta_file.replace('.pkl', '_0.pkl')}")
        """Prepare the dataset by processing each scene."""
        # Process each scene

        for idx, scene_file in enumerate(tqdm(training_scenes)):
            # Process the video
            scene_id = os.path.splitext(scene_file)[0]
            frame_paths = list(sorted([os.path.join(scene_id, i_path) for i_path in os.listdir(os.path.join(self.frames_path, scene_id))]))
            # assert len(frame_paths) == 600

            # Load state information
            state_file = f"{self.states_path}/{scene_id}.jsonl"
            state_data = self._load_state_data(state_file)

            # 600 10

            # Create scene dictionary
            scene_dict = {
                'scene_token': scene_id,  # 场景ID
                'frames': frame_paths,  # 帧图片路径列表
                'speed': [state['vEgo'] for state in state_data],  # 速度序列
                'acceleration': [state['aEgo'] for state in state_data],  # 加速度序列
                'gas': [state['gas'] for state in state_data],  # 油门开度序列
                'gas_pressed': [state['gasPressed'] for state in state_data],  # 油门开度序列
                'steering_angle': [state['steeringAngleDeg'] for state in state_data],  # 方向盘转角序列，左转是正的
                'brake': [state['brake'] for state in state_data],  # 刹车状态序列
                'brake_pressed': [state['brakePressed'] for state in state_data],  # 刹车状态序列
                'gear': [state['gearShifter'] for state in state_data],  # 档位序列
                'left_blinker': [state['leftBlinker'] for state in state_data],  # 左转向灯序列
                'right_blinker': [state['rightBlinker'] for state in state_data],  # 右转向灯序列
                'trajectory': [state['trajectory'] for state in state_data],  # 轨迹序列
                'extrinsic_matrices': [state['extrinsic_matrix'] for state in state_data],  # 外参矩阵序列
                'intrinsic_matrix': [state['intrinsic_matrix'] for state in state_data],  # 内参矩阵序列
                'timestamps': [state['timestamp'] for state in state_data]  # 时间戳序列
            }
            
            self._scenes.append(scene_dict)
        save_pickle_chunks(self.meta_file, self._scenes)

    def _load_state_data(self, state_file: str) -> List[Dict[str, Any]]:
        """Load state data from a JSONL file."""
        state_data = []
        with open(state_file, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                frame_data = next(iter(data.values()))  # Get the data for each frame
                state_data.append(frame_data)
        return state_data
    
    def __len__(self) -> int:
        """Return the number of scenes in the dataset."""
        return len(self._samples)

    def process_item(self, idx: int) -> Dict[str, Any]:
        cur_sample = self._samples[idx]

        with open(self.prompt_file, 'r', encoding='utf-8') as file:
            prompt = file.read()

        temp_info = self.configs
        temp_info.update(cur_sample)
        for placeholder, value in temp_info.items():
            if '__' in placeholder and placeholder in prompt:
                prompt = prompt.replace(placeholder, f'{value:.2f}' if not isinstance(value, str) and not isinstance(value, int) else f'{value}')

        return {
            "messages": [
                {
                    "role": "user",
                    "content":
                        self.format_frames(temp_info['frames']) + \
                        [{"type": "text", "text": prompt}],
                },
                {
                    "role": "assistant",
                    # "content": f'<steering_control>{temp_info["steering_control"]}</steering_control>\n<throttle_and_brake_control>{temp_info["throttle_and_brake_control"]}</throttle_and_brake_control>\n<trajectory>{temp_info["traj_str"]}</trajectory>',
                    "content": f'{temp_info["traj_str"]}',
                }
            ],
            **cur_sample
        }

class D:
    def __init__(self):
        self.ensure_no_overlap = False

# Example usage
if __name__ == "__main__":
    # Initialize the dataset (only load first 10 scenes for demonstration)
    data_args = D()
    dataset = CoVLADataset(
        data_path="/e2e-data/evad-osc-datasets/datasets/CoVLA-Dataset",
        data_args=data_args
    )
