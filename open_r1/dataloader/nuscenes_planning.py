import os
import pickle

from typing import List

import torch
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
from pyquaternion import Quaternion
from collections import defaultdict

from open_r1.dataloader.utils import EgoFutInfo, FrameInfo, EgoLateralMetaActionClassifier, EgoLongitudinalMetaActionClassifier, safe_round, preprocess_meta_action_str, QUERY_PROMPT, load_json
from open_r1.configs import NuScenesDataConfig
from open_r1.dataloader.tools.traj2speed import calculate_velocities
from open_r1.dataloader.tools.speed2acc import calculate_acceleration
from open_r1.dataloader.tools.traj2angle import calculate_steering_angle
from open_r1.dataloader.tools.traj2ka import calculate_trajectory_dynamic_strain_index
from open_r1.dataloader.tools.nav_info import get_nav_info
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from jinja2 import Environment, FileSystemLoader
from open_r1.dataloader.tools.sampler import stratified_sampling
from open_r1.dataloader.tools.traj_bias import compute_trajectory_bias

processor = None

CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]
CAM2VIEW = {
    "CAM_FRONT_LEFT": 'Front-left view',
    "CAM_FRONT": 'Central-front view',
    "CAM_FRONT_RIGHT": 'Front-right view',
    "CAM_BACK_RIGHT": 'Back-right view',
    "CAM_BACK": 'Central-back view',
    "CAM_BACK_LEFT": 'Back-left view'
}

import pickle
import json

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

def mean(l):
    if not isinstance(l, list):
        return l
    if len(l) == 0:
        return 0
    return sum(l) / len(l)


class NuScenesPlanningSupervisedDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        data_args: NuScenesDataConfig
    ):
        super(NuScenesPlanningSupervisedDataset, self).__init__()

        if data_args.force_load or len(data_args.cache_file) > 0 and os.path.isfile(data_args.cache_file):
            print(f'检测到cache file，从 {data_args.cache_file} 中加载 ..')
            self._samples = load_json(data_args.cache_file)
        else:
            self.nusc = NuScenes(version=data_args.version, dataroot=data_path, verbose=False)
            self.scene_names = set(splits.create_splits_scenes()[data_args.split])
            self.dataroot = data_path
            self.obs_len = data_args.obs_len
            self.fut_len = data_args.fut_len
            self.meta_file = os.path.join(data_args.meta_dir, f"{data_args.split}.pkl")
            os.makedirs(data_args.meta_dir, exist_ok=True)
            self.rebuild = data_args.rebuild
            self.system_prompt = None
            self.lon_classifier = EgoLongitudinalMetaActionClassifier()
            self.lat_classifier = EgoLateralMetaActionClassifier()
            self.num_path_points = data_args.num_path_points
            self.path_point_distance = data_args.path_point_distance
            self.decimal = data_args.decimal

            self.prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", f'{data_args.query_prompt_type}.txt')

            # prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
            # jinja2_env = Environment(loader=FileSystemLoader(prompts_dir))
            # self.template = jinja2_env.get_template(f"{data_args.query_prompt_type}.j2")

            self.required_cams = data_args.required_cams.split(',')
            self.image_description = data_args.image_description

            self.resize_kwargs = {
                "resized_height": 448,
                "resized_width": 896
            }

            if os.path.exists(self.meta_file):
                print(f'找到：{self.meta_file} 开始读取..')
                self._samples = pickle.load(open(self.meta_file, "rb"))
            else:
                print(f'未找到：{self.meta_file}')
                self.preload()
            self._samples = [self.format_single_sample(cur_info) for cur_info in self._samples]
            print(f'保存到cache file {data_args.cache_file} ..')
            save_json(data_args.cache_file, self._samples)

        # cold start
        if len(data_args.sft_indices) > 0:
            if '//except' not in data_args.sft_indices:
                sft_indices = load_json(data_args.sft_indices)
                self._samples = [i for i in self._samples if i['sample_token'] in sft_indices]
            else:
                sft_indices = load_json(data_args.sft_indices.replace('//except', ''))
                self._samples = [i for i in self._samples if i['sample_token'] not in sft_indices]

        if data_args.sample_size > 0:
            assert len(data_args.sample_ratios.split(',')) == len(data_args.sample_bins.split(',')) - 1, f"{len(data_args.sample_ratios.split(','))}, {len(data_args.sample_bins.split(','))}"
            self._samples = stratified_sampling(self._samples, [float(i) for i in data_args.sample_bins.split(',')], [int(i) for i in data_args.sample_ratios.split(',')], data_args.sample_size)

        if torch.cuda.current_device() == 0:
            print(f"Final Data Length: {len(self._samples)}")
            print("="*40 + "\n")


    def preload(self):
        _samples = []
        for scene in tqdm(self.nusc.scene):
            name = scene["name"]
            if name not in self.scene_names:
                continue
            infos = self.process_single_scene(scene["token"])
            if len(infos) > 0:
                _samples.extend(infos)
        save_pickle(self.meta_file, _samples)
        print('保存', self.meta_file)
        self._samples = _samples

    # TODO: visualize to check the correctness of trajectory
    def process_single_scene(self, scene_token) -> List[dict]:
        scene_info = self.nusc.get("scene", scene_token)
        current_sample_token = scene_info["first_sample_token"]
        last_sample_token = scene_info["last_sample_token"]

        # load sample infos of the whole scene
        samples = []
        while current_sample_token != last_sample_token:
            sample_info = self.nusc.get("sample", current_sample_token)
            if sample_info is not None:
                samples.append(sample_info)

            if current_sample_token == last_sample_token:
                break
            current_sample_token = sample_info["next"]

        # sliding window
        infos = []
        for idx in range(self.obs_len - 1, len(samples) - self.fut_len):
            current = samples[idx]
            slices = samples[idx - self.obs_len + 1 : idx + self.fut_len + 1]

            coords = []
            for i, sample in enumerate(slices):
                coord = self.get_world_coord(sample)
                coords.append(coord)
            coords = np.array(coords)

            frames = [] # list of dict
            for i, sample in enumerate(slices[:self.obs_len]):
                cam2frame = {}
                for cam in CAMS:
                    frame_info = self.nusc.get("sample_data", sample['data'][cam])
                    cam2frame[cam] = frame_info['filename']
                frames.append(cam2frame)

            rotation_matrix = self.get_rotation_matrix(current)
            origin = coords[self.obs_len - 1, :]
            trajectory = np.dot(np.array(coords) - origin, rotation_matrix.T)[:, :2]
            speed = (
                np.linalg.norm(coords[self.obs_len] - coords[self.obs_len - 2]) / 1.0
            )
            velocities = np.zeros_like(coords)
            velocities = coords[1:] - coords[:-1]
            velocities[0] = velocities[1]

            speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
            lon_action = self._calculate_lon_action(trajectory, speeds, is_obs=False)
            lat_action = self._calculate_lat_action(trajectory, speeds, is_obs=False)

            obs_lon_action = self._calculate_lon_action(trajectory, speeds, is_obs=True)
            obs_lat_action = self._calculate_lat_action(trajectory, speeds, is_obs=True)

            # navigation command
            ego_fut_trajs = trajectory[self.obs_len:]
            if ego_fut_trajs[-1][1] >= 2:
                command = "Turn Right"
            elif ego_fut_trajs[-1][1] <= -2:
                command = "Turn Left"
            else:
                command = "Go Straight"

            infos.append(
                {
                    "scene_token": scene_token,
                    "sample_token": current["token"],
                    "frames": frames,
                    "speed": speed,
                    "obs_trajectory": trajectory[: self.obs_len],
                    "fut_trajectory": trajectory[self.obs_len :],
                    "lat_action": lat_action,
                    "lon_action": lon_action,
                    "obs_lat_action": obs_lat_action,
                    "obs_lon_action": obs_lon_action,
                    # "navigation_path": navigation_info["path"],
                    # "navigation_valid": navigation_info["valid"],  # 有效性标记
                    "navi_command": command,
                    "origin": origin,
                }
            )

        return infos

    def _calculate_lon_action(self, trajectory, speeds, is_obs):
        """计算纵向meta action"""
        class EgoFutInfo:
            def __init__(self, trajs, velocity):
                self.trajs = trajs
                self.velocity = velocity
                self.masks = np.ones(len(trajs))

        class MetaData:
            def __init__(self, fut_info):
                self.ego_fut_info = fut_info
        if is_obs == True:
            trajs = trajectory[:self.obs_len]
            velocity = speeds[:self.obs_len]
        else:
            trajs = trajectory[self.obs_len:]
            velocity = speeds[self.obs_len-1:]
            
        lon_result = self.lon_classifier.process(
            MetaData(EgoFutInfo(trajs, velocity))
        )
        result = lon_result['category_str']
        return result.split("_")[-1]

    def _calculate_lat_action(self, trajectory, speeds, is_obs):
        """计算横向meta action"""
        if is_obs == True:
            lat_fut_info = EgoFutInfo(
                    trajs=trajectory[:self.obs_len],
                    velocity=speeds[:self.obs_len],
                    masks=np.ones(len(trajectory[:self.obs_len]), dtype=bool)  # 全有效
                )
        else:
            lat_fut_info = EgoFutInfo(
                    trajs=trajectory[self.obs_len:],  # future trajectory (N,2)
                    velocity=speeds[self.obs_len:],  # speed (N,1)
                    masks=np.ones(len(trajectory[self.obs_len:]), dtype=bool)  # 全有效
                )
        lat_result = self.lat_classifier.process(FrameInfo(lat_fut_info))
        result = lat_result["category_str"]
        return result.split("_")[-1]

    @staticmethod
    def interp_arc(n: int, points: np.ndarray):
        if len(points) < 2:
            return points
        # 计算累积弧长
        dists = np.linalg.norm(points[1:] - points[:-1], axis=1)
        cum_dists = np.concatenate([[0], np.cumsum(dists)])
        total_dist = cum_dists[-1]
        if total_dist == 0:
            return np.zeros((n, 2))
        # 创建插值函数
        fx = interp1d(cum_dists, points[:, 0], kind='linear')
        fy = interp1d(cum_dists, points[:, 1], kind='linear')
        s_new = np.linspace(0, total_dist, n)
        return np.column_stack([fx(s_new), fy(s_new)])

    def get_world_coord(self, sample):
        lidar = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        pose = self.nusc.get("ego_pose", lidar["ego_pose_token"])
        return pose["translation"]

    def get_rotation_matrix(self, sample):
        lidar = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego2global = self.nusc.get("ego_pose", lidar["ego_pose_token"])
        lidar2ego = self.nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"])
        l2e_r = Quaternion(lidar2ego["rotation"]).rotation_matrix
        e2g_r = Quaternion(ego2global["rotation"]).rotation_matrix
        g2l_r = np.linalg.inv(l2e_r @ e2g_r)
        return g2l_r

    def get_info(self, idx):
        return self._samples[idx]

    def __len__(self):
        return len(self._samples)

    def format_frames(self, frames, required_cams):
        return sum(
            [[{"type": "text", "text": f"{CAM2VIEW[i_cam]}: "}] + [{"type": "image", "image": f"file://{os.path.join(self.dataroot, frames[i_cam])}", **self.resize_kwargs}] for i_cam in required_cams],
            []
        )

    def format_raw_frames(self, frames, required_cams):
        return sum(
            [[{"type": "image", "image": f"file://{os.path.join(self.dataroot, frames[i_cam])}", **self.resize_kwargs}] for i_cam in required_cams],
            []
        )
    def format_single_sample(self, info):
        with open(self.prompt_file, 'r', encoding='utf-8') as file:
            prompt = file.read()
        
        def trans_traj(l_traj):
            return [[i[1], -i[0]] for i in l_traj]

        full_traj = trans_traj(info['obs_trajectory'].tolist()) + trans_traj(info['fut_trajectory'].tolist())

        full_v = calculate_velocities(full_traj, time_interval=0.5)
        info['__speed__'] = full_v[self.obs_len-2]
        full_acc = calculate_acceleration(full_v, delta=0.5)
        info['__acceleration__'] = full_acc[self.obs_len-3]
        steering_angle_list = calculate_steering_angle(trans_traj(full_traj))
        info['__steering_degree__'] = steering_angle_list[self.obs_len-2]

        prompt = prompt.replace('__steering_degree__', f"{info['__steering_degree__']:.2f}")
        prompt = prompt.replace('__speed__', f"{info['__speed__']:.2f}")
        prompt = prompt.replace('__acceleration__', f"{info['__acceleration__']:.2f}")

        ka_fut = calculate_trajectory_dynamic_strain_index(full_traj[self.obs_len:])
        avg_angle = sum(steering_angle_list[self.obs_len-1:]) / len(steering_angle_list[self.obs_len-1:])
        info['__nav_info__'] = get_nav_info(avg_angle, ka_fut)
        prompt = prompt.replace('__nav_info__', f"{info['__nav_info__']}")

        traj_str = [f"[{safe_round(x, self.decimal)}, {safe_round(y, self.decimal)}]" for (x, y) in trans_traj(info['fut_trajectory'])]
        traj_str = '[' + ', '.join(traj_str) + ']'
        gt_str = traj_str

        trajectory_bias, extended_trajectory, extended_trajectory_gt = compute_trajectory_bias(full_traj, self.obs_len, len(full_traj) - self.obs_len, 2)

        if len(self.required_cams) == 1:
            # 如果只有单目图片，就全部拼在前面
            image_prompt = sum(
                [
                    self.format_raw_frames(cam_frames, self.required_cams)
                    for mom_idx, cam_frames in enumerate(info['frames'])
                ], []
            )
        else:
            raise NotImplementedError
            # # 如果有多目图片，就要说明该图片是哪一目
            # image_prompt = sum(
            #     [
            #         [{"type": "text", "text": f"Moment {mom_idx+1}: "}] + self.format_frames(cam_frames, self.required_cams) + [{"type": "text", "text": "\n"}]
            #         for mom_idx, cam_frames in enumerate(info['frames'])
            #     ], []
            # )
        
        # image_description = self.image_description.replace('__frame_length__', f"{len(info['frames'])+1}")
        return {
            "messages": [
                {
                    "role": "user",
                    "content": image_prompt + \
                        [{"type": "text", "text": prompt}],
                },
                {
                    "role": "assistant",
                    "content": gt_str
                }
            ],
            'trajectory': full_traj,
			"navi_command": info.get("navi_command"),
            "speed": info['__speed__'],
            "trajectory_bias": trajectory_bias,
            'trajectory_bias_pred': extended_trajectory,
            'trajectory_bias_gt': extended_trajectory_gt,
            'gold': {
                'steering_angle_list': steering_angle_list,
                'speed_list': full_v,
                'ka': calculate_trajectory_dynamic_strain_index(full_traj),
                'ka_obs': calculate_trajectory_dynamic_strain_index(full_traj[:self.obs_len]),
                'ka_fut': ka_fut
            }
            # "id": f"{info['']}_{info['']}
        }

    # def get_is_ego_baseline_msg(self, idx):
    #     info = self._samples[idx]

    #     obs_traj = "["  + ",".join([f"[{safe_round(x, self.decimal)}, {safe_round(y, self.decimal)}]" for (x, y) in info['obs_trajectory']]) + "]"
    #     prompt_info = {
    #         "speed": f"{safe_round(info['speed'], self.decimal)}",
    #         "historical_trajectory": obs_traj,
    #         "navi_command": info.get("navi_command")
    #     }

    #     fut_traj = "["  + ",".join([f"[{safe_round(x, self.decimal)}, {safe_round(y, self.decimal)}]" for (x, y) in info['fut_trajectory']]) + "]"

    #     return {
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": 
    #                     [{"type": "image", "image": f"file://{os.path.join(self.dataroot, image_path)}", **self.resize_kwargs} for image_path in info['frames']] + \
    #                     [{"type": "text", "text": self.template.module.prompt_template(**prompt_info)}],
    #             },
    #             {
    #                 "role": "assistant",
    #                 "content": self.template.module.response_template(trajectory=fut_traj),
    #             }
    #         ]
    #     }
