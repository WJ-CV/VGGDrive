import os
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


import pickle
import json
import random

from PIL import Image

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # if isinstance(data, list):
    #     num_samples = len(data)
    #     num_selected = int(num_samples * 0.8)
    #     selected_data = random.sample(data, num_selected)  # 随机选择50%的数据
    #     return selected_data

    return data


def save_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

class NAVSIM(Dataset):
    def __init__(
        self,
        data_path,
        data_args
    ):
        self.data_args = data_args
        if data_args.force_load or len(data_args.cache_file) > 0:
            print(f'检测到cache file,从 {data_args.cache_file} 中加载 ..')
            self._samples = []
            for cur_cache in data_args.cache_file.split(';'):
                self._samples.extend(load_json(cur_cache))
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        return self._samples[index]
