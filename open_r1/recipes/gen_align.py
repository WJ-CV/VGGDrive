# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import importlib
from collections import defaultdict
from pathlib import Path
from functools import partial
from copy import deepcopy
# from torch.utils.data import Subset

import datasets
import torch
import numpy as np
import random
import transformers
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from typing import Optional
from open_r1.configs import SFTConfig, NuScenesDataConfig, preprocess_data_args, index_data_args
from open_r1.utils.callbacks import get_callbacks
import open_r1.dataloader as DATASETS
from open_r1.dataloader.utils import save_json, load_json

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import field
from qwen_vl_utils import process_vision_info
logger = logging.getLogger(__name__)
from dataclasses import dataclass

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward

processor = None

def collate_fn(examples, prompt_mask):
    texts = [
        processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
        for example in examples
    ]
    image_inputs = []
    for example in examples:
        imgs, vids = process_vision_info(example["messages"])
        image_inputs.append(imgs)
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    if not prompt_mask:
        return batch

    # 在label中不带Prompt
    question_texts = [
        processor.apply_chat_template([example["messages"][0]], tokenize=False, add_generation_prompt=True)
        for example in examples
    ]

    for idx, i_question_text in enumerate(question_texts):
        cur_question_batch = processor(
            text=[i_question_text],
            images=image_inputs,
            return_tensors="pt",
            padding=False,
        )
        batch["labels"][idx, :cur_question_batch['input_ids'].shape[1]] = -100

    return batch

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, action_type, samples_per_type, dataset_list, ratio_shuffle, mixed_shuffle):
        self.datasets = dataset_list
        self.action_type = action_type
        self.ratio_shuffle = ratio_shuffle
        self.mixed_shuffle = mixed_shuffle
        self._samples = []
        for dataset in self.datasets:
            self._samples.extend(dataset._samples)

        self.samples_per_type = samples_per_type
        self.update_action_indices()

        if torch.cuda.current_device() == 0:
            print("\n=== Base Data Distribution Statistics ===")
            print(f"Total samples: {len(self._samples)}")
            print(f"Without actions: {len(self.wo_action_indices)}")
            print(f"Number of action classes: {self.num_classes}")
            print("Samples per class:")
            for cur_action in sorted(self.actions):
                print(f"  {cur_action}: {len(self.action_indices[cur_action])}")
            print("="*40 + "\n")

        if self.ratio_shuffle:
            self.local_shuffle()
        else:
            self.local_fixed()

        if self.mixed_shuffle:
            random.shuffle(self.shuffled_samples)
        self.print_shuffled_samples()

    def print_shuffled_samples(self):
        print(f"Current Data Length: {len(self.shuffled_samples)}")
        lat_action_list = [i_sample[self.action_type] for i_sample in self.shuffled_samples if self.action_type in i_sample.keys()]
        print(f"Samples per class [{self.action_type}]:")
        for cur_action in set(lat_action_list):
            print(f"  {cur_action}: {lat_action_list.count(cur_action)}")
        print('Sample 0 & 1 & 2')
        print([self.shuffled_samples[0]])
        print("="*40 + "\n")
        print([self.shuffled_samples[1]])
        print("="*40 + "\n")
        print([self.shuffled_samples[2]])
        print("="*80 + "\n")

    def update_action_indices(self):
        self.action_indices, self.wo_action_indices = defaultdict(list), []
        for idx in range(len(self._samples)):
            if self.action_type in self._samples[idx].keys():
                cur_action = self._samples[idx][self.action_type]
                self.action_indices[cur_action].append(idx)
            else:
                self.wo_action_indices.append(idx)

        self.actions = list(self.action_indices.keys())
        self.num_classes = len(self.actions)

    def local_shuffle(self):
        print('-' * 80)
        print('In local_shuffle ..')
        self.update_action_indices()
        sampled_indices = []
        for cur_action in self.actions:
            indices = self.action_indices[cur_action]
            if len(indices) >= self.samples_per_type[cur_action]:
                selected = np.random.choice(indices, self.samples_per_type[cur_action], replace=False)
            else:
                selected = np.random.choice(indices, self.samples_per_type[cur_action], replace=True)
            sampled_indices.extend(selected.tolist())

        np.random.shuffle(self.wo_action_indices)
        np.random.shuffle(sampled_indices)
        self.shuffled_samples = [self._samples[i] for i in self.wo_action_indices + sampled_indices]

    def local_fixed(self):
        print('-' * 80)
        print('In local_fixed ..')
        self.shuffled_samples = self._samples

    def __len__(self):
        return len(self.shuffled_samples)

    def __getitem__(self, idx):
        return self.shuffled_samples[idx]





import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

def draw_trajectory(trajectory_data, cur_data, trajectory_bias_pred, output_path):
    """绘制轨迹并保存为图片"""
    # 提取x和y坐标
    points = np.array(trajectory_data)
    points_pred = np.array(trajectory_bias_pred)
    points_gt = np.array(cur_data['trajectory'])
    x = points[:, 0]
    y = points[:, 1]

    avg_angle = sum(cur_data['gold']['steering_angle_list'][3:]) / len(cur_data['gold']['steering_angle_list'][3:])

    # 创建绘图
    plt.figure(figsize=(13, 6))
    plt.plot(x, y, 'ro-')  # 蓝色圆点连线
    plt.plot(points_pred[:, 0], points_pred[:, 1], 'bo-')
    plt.plot(points_gt[:, 0], points_gt[:, 1], 'go-')
    plt.title('Vehicle Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f"{avg_angle:.2f}_"  + f"{cur_data['gold']['ka']:.2f}_" + f"{cur_data['gold']['ka_obs']:.2f}_" + f"{cur_data['gold']['ka_fut']:.2f}")
    plt.xlim([-15, 30])
    plt.ylim([-8, 8])
    plt.grid(True)
    
    # 保存图像
    plt.savefig(output_path)
    plt.close()


def main(script_args, data_args, sft_args, model_args):
    # Set seed for reproducibility
    set_seed(sft_args.seed)

    data_args.split = 'val'
    cur_save_dataset_name = os.path.split(data_args.cache_file)[-1].replace('.json', '')

    print('sft_args.output_dir rrrrrrrrrrrrrrrrrrrrrrrr', sft_args.output_dir)
    data_args = preprocess_data_args(data_args)

    base_path = os.path.split(script_args.dataset_name)[0]
    test_json_save_path = '/e2e-data/users/lg/workspace/users/zhangyikai/vla_source/data'
    cur_dataset_name = f'{cur_save_dataset_name}_{data_args.query_prompt_type}'

    saved_args = {
        'model_name_or_path': model_args.model_name_or_path,
        'notes': sft_args.notes,
        'learning_rate': sft_args.learning_rate,
        'weight_decay': sft_args.weight_decay,
        'dataset_name': script_args.dataset_name,
        'dataset_classname': data_args.dataset_classname,
        'per_device_train_batch_size': sft_args.per_device_train_batch_size,
        'warmup_ratio': sft_args.warmup_ratio,
        'ratio_shuffle': data_args.ratio_shuffle,
        'mixed_shuffle': data_args.mixed_shuffle,
        'required_cams': data_args.required_cams,
        'image_description': data_args.image_description,
        'balanced_action': data_args.balanced_action,
        'query_prompt_type': data_args.query_prompt_type,
    }
    print(saved_args)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = sft_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {sft_args.local_rank}, device: {sft_args.device}, n_gpu: {sft_args.n_gpu}"
        + f" distributed training: {bool(sft_args.local_rank != -1)}, 16-bits training: {sft_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"NuScenes parameters {data_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Sft parameters {sft_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(sft_args.output_dir):
        last_checkpoint = get_last_checkpoint(sft_args.output_dir)
    if last_checkpoint is not None and sft_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")


    ################
    # Load datasets
    ################
    dataset_class = getattr(DATASETS, data_args.dataset_classname)
    cur_data_args = index_data_args(deepcopy(data_args), 0)
    train_dataset = dataset_class(data_path=script_args.dataset_name, data_args=cur_data_args)

    # 创建输出目录
    output_base_dir = f"./trajectory_data_{data_args.dataset_classname}"
    os.makedirs(output_base_dir, exist_ok=True)

    for idx in range(len(train_dataset._samples)):
        if idx > 100:
            assert 0
        cur_data = train_dataset._samples[idx]
        
        # 为每个样本创建单独的文件夹
        sample_dir = output_base_dir
        
        avg_angle = sum(cur_data['gold']['steering_angle_list'][3:]) / len(cur_data['gold']['steering_angle_list'][3:])
        # 保存所有图片帧
        img_count = 0
        for message in cur_data['messages']:
            if message['role'] == 'user':
                for content in message['content']:
                    if content['type'] == 'image':
                        src_path = content['image'].replace('file://', '')
                        img_name = f"{avg_angle:.2f}_{cur_data['gold']['ka']:.2f}_{cur_data.get('trajectory_bias', 'N/A')}_{idx}_{img_count}.jpg"
                        dst_path = os.path.join(sample_dir, img_name)
                        shutil.copy(src_path, dst_path)
                        img_count += 1


        # 绘制并保存轨迹图
        for message in cur_data['messages']:
            if message['role'] == 'assistant':
                # try:
                # 解析轨迹数据
                trajectory_data = eval(message['content'])
                traj_output_path = os.path.join(sample_dir, f"{avg_angle:.2f}_{cur_data['gold']['ka']:.2f}_{cur_data.get('trajectory_bias', 'N/A'):.2f}_{idx}_trajectory.png")
                draw_trajectory(trajectory_data, cur_data, cur_data['trajectory_bias_pred'], traj_output_path)
                # except:
                #     print(f"Error processing trajectory for sample {idx}")
        
        # 保存元数据
        # with open(os.path.join(sample_dir, f"{idx}_metadata.txt"), 'w') as f:
        #     f.write(f"Navigation Command: {cur_data.get('navi_command', 'N/A')}\n")
        #     f.write(f"Speed: {cur_data.get('speed', 'N/A')} m/s\n")
        #     f.write(f"Trajectory Bias: {cur_data.get('trajectory_bias', 'N/A')}\n")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, NuScenesDataConfig, SFTConfig, ModelConfig))
    script_args, data_args, sft_args, model_args = parser.parse_args_and_config()
    main(script_args, data_args, sft_args, model_args)
