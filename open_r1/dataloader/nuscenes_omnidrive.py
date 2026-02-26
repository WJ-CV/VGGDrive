import os
import pickle
import random

from typing import List

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
from pyquaternion import Quaternion

from open_r1.dataloader.utils import EgoFutInfo, FrameInfo, EgoLateralMetaActionClassifier, EgoLongitudinalMetaActionClassifier, safe_round, preprocess_meta_action_str, QUERY_PROMPT, load_json
from open_r1.configs import NuScenesDataConfig
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
import json


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

class NuScenesOmniDrive(Dataset):
    def __init__(
        self,
        data_path: str,
        data_args: NuScenesDataConfig
    ):
        super(NuScenesOmniDrive, self).__init__()
        self.nusc = NuScenes(version=data_args.version, dataroot=data_path, verbose=False)
        self.scene_names = set(splits.create_splits_scenes()[data_args.split])
        self.split = data_args.split
        self.dataroot = data_path
        self.meta_file = os.path.join(data_args.meta_dir, f"omnidrive_{data_args.split}.pickle")
        os.makedirs(data_args.meta_dir, exist_ok=True)
        self.rebuild = data_args.rebuild
        self.vqa_root = data_args.omnidrive_vqa_folder
        self.decimal = data_args.decimal

        self.resize_kwargs = {
            "resized_height": 448,
            "resized_width": 896
        }

        if os.path.exists(self.meta_file) and not self.rebuild:
            self._samples = pickle.load(open(self.meta_file, "rb"))
        else:
            self.preload()

        # self._samples = random.sample(self._samples, 10000)
        print(f"OmniDrive Data Length: {len(self._samples)}")

    def preload(self):
        _samples = []
        for sample in tqdm(self.nusc.sample):
            vqas = self.process_single_sample(sample)
            if len(vqas) > 0:
                _samples.extend(vqas)
        pickle.dump(_samples, open(self.meta_file, "wb"))
        self._samples = _samples

    def process_single_sample(self, info):
        print(info.keys())
        print('prev', info['prev'])
        print('next', info['next'])
        print('scene_token', info['scene_token'])
        scene_info = self.nusc.get("scene", info['scene_token'])
        current_sample_token = scene_info["first_sample_token"]
        last_sample_token = scene_info["last_sample_token"]

        # load sample infos of the whole scene
        samples = []
        sample_tokens = []
        while current_sample_token != last_sample_token:
            print('c', current_sample_token)
            sample_info = self.nusc.get("sample", current_sample_token)
            if sample_info is not None:
                samples.append(sample_info)
                sample_tokens.append(current_sample_token)

            if current_sample_token == last_sample_token:
                break
            current_sample_token = sample_info["next"]
        sample_tokens.index(info['token'])
        print(info['token'], 'info_token')
        assert 0
        frames = [] # list of dict
        slices = samples[-self.obs_len:]
        for i, sample in enumerate(slices[:self.obs_len]):
            cam2frame = {}
            for cam in CAMS:
                frame_info = self.nusc.get("sample_data", sample['data'][cam])
                cam2frame[cam] = frame_info['filename']
            frames.append(cam2frame)
        cam_names = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
        cam_prompt = []
        for cname in cam_names:
            cam_info = self.nusc.get("sample_data", info['data'][cname])
            # cpath =  f"file://{os.path.join(self.dataroot, cam_info['filename'])}"
            cpath =  f"{os.path.join(self.dataroot, cam_info['filename'])}"
            cam_prompt.extend([
                # {"type": "text", "text": f"{cname}\n"},
                {"type": "image", "image": f"{cpath}", **self.resize_kwargs},
                # {"type": "text", "text": "\n"},
            ])
        cam_prompt.extend([
            {"type": "text", "text": f"The six images were captured from the front-left, front, front-right, back-left, back, and back-right cameras. "},
        ])

        vqas = list()
        vqas.extend(self.process_conv(info['token'], cam_prompt))
        vqas.extend(self.process_desc(info['token'], cam_prompt))
        vqas.extend(self.process_keywords(info['token'], cam_prompt))
        vqas.extend(self.process_vqa(info['token'], cam_prompt))

        return vqas

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]

    def process_conv(self, token, cam_prompt):
        try:
            with open(os.path.join(self.vqa_root, f"conv/{self.split}/{token}.json"), 'r') as ifp:
                # [{'question': 'xx', 'answer': xx}, ...]
                js = json.load(ifp)
        except FileNotFoundError:
            # print(os.path.join(self.vqa_root, f"conv/{self.split}/{token}.json"))
            return []

        samples = list()
        for sample in js:
            samples.append({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            *cam_prompt,
                            {"type": "text", "text": sample['question']},
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": sample['answer']
                    }
                ]
            })
        return samples

    def process_desc(self, token, cam_prompt):
        try:
            with open(os.path.join(self.vqa_root, f"desc/{self.split}/{token}.json"), 'r') as ifp:
                # {'description': 'xx', 'action': xx}
                js = json.load(ifp)
        except FileNotFoundError:
            # print(os.path.join(self.vqa_root, f"desc/{self.split}/{token}.json"))
            return []

        return [{
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *cam_prompt,
                        {"type": "text", "text": js['description']},
                    ]
                },
                {
                    "role": "assistant",
                    "content": js['action']
                }
            ]
        }]

    def process_keywords(self, token, cam_prompt) -> List:
        try:
            with open(os.path.join(self.vqa_root, f"keywords/{self.split}/{token}.json"), 'r') as ifp:
                # 'interpretative caption'
                js = json.load(ifp)
        except FileNotFoundError:
            # print(os.path.join(self.vqa_root, f"keywords/{self.split}/{token}.json"))
            return []

        prompt = ("Suppose you are an experienced driver. "
            "Please describe your driving decisions based "
            "on the information from the surrounding cameras."
        )
        return [{
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *cam_prompt,
                        {"type": "text", "text": prompt},
                    ]
                },
                {
                    "role": "assistant",
                    "content": js
                }
            ]
        }]

    def process_vqa(self, token, cam_prompt):
        try:
            with open(os.path.join(self.vqa_root, f"vqa/{self.split}/{token}.json"), 'r') as ifp:
                # [{'question': 'xx', 'answer': xx}, ...]
                js = json.load(ifp)
        except FileNotFoundError:
            # print(os.path.join(self.vqa_root, f"vqa/{self.split}/{token}.json"))
            return []

        samples = list()
        for sample in js:
            samples.append({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            *cam_prompt,
                            {"type": "text", "text": sample['question']},
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": sample['answer']
                    }
                ]
            })
        return samples
