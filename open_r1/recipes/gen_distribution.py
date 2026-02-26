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
from tqdm import tqdm
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
import seaborn as sns
from copy import deepcopy

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

import matplotlib.pyplot as plt
from collections import Counter
import os

def plot_and_save_distributions(data_dict, prefix, output_dir='distributions'):
    """
    绘制字典中每个键对应列表值的分布图，并保存为图片
    
    参数:
        data_dict (dict): 键为字符串，值为字符串列表的字典
        output_dir (str): 保存图片的目录，默认为'distributions'
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for key, values in data_dict.items():
        if not values:  # 跳过空列表
            continue
            
        # 计算每个值的出现频率
        value_counts = Counter(values)
        labels, counts = zip(*value_counts.most_common())
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        # 绘制条形图
        bars = plt.bar(labels, counts)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # 设置图表标题和标签
        plt.title(f'Distribution of {key}')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')  # 旋转x轴标签
        
        # 调整布局防止标签被截断
        plt.tight_layout()
        
        # 保存图表
        filename = f"{key.replace('/', '_')}.png"  # 替换可能造成问题的字符
        filepath = os.path.join(output_dir, f'{prefix}_{filename}')
        plt.savefig(filepath)
        plt.close()  # 关闭图形以释放内存
        
        print(f"Saved distribution plot for '{key}' to {filepath}")

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
import numpy as np
import os
from math import ceil

def plot_float_distribution(data, save_filename, global_min=None, global_max=None, output_dir='distributions', bins='auto', chunk_size=10000):
    """
    绘制float列表的分布图并保存为图片，并在每个柱形上标注样本数量
    适用于大数据量情况，采用分块处理
    
    参数:
    data -- 包含float的列表或numpy数组
    save_filename -- 要保存的图片文件名(可以包含路径)
    output_dir -- 输出目录
    bins -- 直方图的bin数量，默认为'auto'
    chunk_size -- 每个分块的大小，默认为100000
    """
    plt.figure(figsize=(10, 6))
    
    data = np.asarray(data)
    total_samples = len(data)
    
    # 确定全局的bin边界
    if isinstance(bins, (int, np.integer, str)):
        # 如果是自动或固定数量bin，先计算全局范围
        global_min = np.min(data) if global_min is None else global_min
        global_max = np.max(data) if global_max is None else global_max
        bins = 100

        # 创建全局统一的bin边界
        bin_edges = np.linspace(global_min, global_max, bins + 1)
    else:
        # 如果已经提供了bin边界，直接使用
        bin_edges = np.asarray(bins)

    
    # 初始化总计数
    total_counts = np.zeros(len(bin_edges) - 1)
    
    # 分块处理数据
    for i in range(0, total_samples, chunk_size):
        chunk = data[i:i + chunk_size]
        chunk_counts, _ = np.histogram(chunk, bins=bin_edges)
        total_counts += chunk_counts

    saved_json_data = {
        'global_max': global_max,
        'total_counts': f'{total_counts}',
        'bin_edges': f'{bin_edges}',
        'length': f'{sum(total_counts)}'
    }
    # 绘制直方图
    patches = plt.bar(
        bin_edges[:-1], 
        total_counts, 
        width=np.diff(bin_edges), 
        edgecolor='black', 
        alpha=0.7,
        align='edge'
    )
    
    # 在每个柱形上方标注频数（只标注较大的柱形）
    max_count = max(total_counts)
    for bin_edge, count, patch in zip(bin_edges[:-1], total_counts, patches):
        if count > 0.01 * max_count:  # 只标注超过最大计数1%的柱形
            plt.text(
                bin_edge + (bin_edges[1] - bin_edges[0]) / 2,  # x 位置：柱形中心
                count + 0.02 * max_count,                      # y 位置：略高于柱形顶部
                f'{int(count)}',                              # 显示的文本（频数）
                ha='center', va='bottom',                     # 水平居中，垂直底部对齐
                fontsize=8 if total_samples > 1e6 else 10     # 大数据量使用较小字体
            )
    
    # 添加标题和标签
    plt.title(f'Distribution of Float Values (n={total_samples:,})', fontsize=15)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # 添加网格线
    plt.grid(axis='y', alpha=0.75)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图片
    plt.savefig(os.path.join(output_dir, f'{save_filename}_{global_min}_{global_max}.png'), bbox_inches='tight', dpi=300)
    save_json(os.path.join(output_dir, f'{save_filename}_{global_min}_{global_max}.json'), saved_json_data)
    plt.close()
    
    print(f"Distribution plot saved as {os.path.join(output_dir, f'{save_filename}.png')}")

def plot_and_save_heatmap(data_pairs, steering_classes, longitudinal_classes, file_name, output_dir='distributions'):
    count_matrix = np.zeros((len(steering_classes), len(longitudinal_classes)), dtype=int)

    print('In plot_and_save_heatmap')
    for steer_class, long_class in data_pairs:
        row = steering_classes.index(steer_class)
        col = longitudinal_classes.index(long_class)

        count_matrix[row, col] += 1

    row_sums = count_matrix.sum(axis=1).reshape(-1, 1)
    col_sums = count_matrix.sum(axis=0)
    total_sum = count_matrix.sum()

    matrix_with_sums = np.concatenate([count_matrix, row_sums], axis=1)
    matrix_with_sums = np.concatenate([matrix_with_sums, np.append(col_sums, total_sum).reshape(1, -1)], axis=0)

    steering_classes_with_sum = steering_classes
    longitudinal_classes_with_sum = longitudinal_classes

    # 绘制热力图
    plt.figure(figsize=(12, 9))
    sns.heatmap(count_matrix, 
                annot=True, 
                fmt='d',
                cmap='YlOrRd',
                xticklabels=longitudinal_classes_with_sum,
                yticklabels=steering_classes_with_sum,
                linewidths=0.5)

    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, f'{file_name}_heatmap.png'), dpi=300, bbox_inches='tight')

def main(script_args, data_args, sft_args, model_args):
    # Set seed for reproducibility
    set_seed(sft_args.seed)

    print('sft_args.output_dir rrrrrrrrrrrrrrrrrrrrrrrr', sft_args.output_dir)

    data_args = preprocess_data_args(data_args)
    base_path = os.path.split(script_args.dataset_name)[0]

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
    train_dataset_list = [dataset_class(data_path=script_args.dataset_name, data_args=cur_data_args)]

    train_dataset = CombinedDataset(
        action_type=data_args.balanced_action,
        samples_per_type={
            'Go Straight': 8192,
            'Turn Left': 2048,
            'Turn Right': 2048,
            'Deviate Left': 2048,
            'Deviate Right': 2048,
            'U Turn': 512
        },
        dataset_list=train_dataset_list,
        ratio_shuffle=data_args.ratio_shuffle,
        mixed_shuffle=data_args.mixed_shuffle
    )
    def get_images(content):
        return [i['image'].replace('file:///e2e-data/evad-osc-datasets/datasets/', '') for i in content if i['type'] == 'image']
    # def get_question(content):
    #     for i in content:
    #         if i['type'] == 'text':
    #             return i['text']
    #     return None
    # def get_question(content):
    #     q_prompt = ''
    #     image_idx = 0
    #     for i in content:
    #         if i['type'] == 'text':
    #             q_prompt += i['text']
    #         elif i['type'] == 'image':
    #             q_prompt += f'<image {image_idx}>'
    #             image_idx += 1
    #     return q_prompt

    def get_question(content):
        for i in range(len(content['content'])):
            if content['content'][i]["type"] == 'image':
                content['content'][i]['image'] = content['content'][i]['image'].replace(f'file://{base_path}/', '')
        return [content]
    
    results = []

    from collections import defaultdict
    distributed_dict = defaultdict(list)
    print('Dataset size: ', len(train_dataset._samples))
    data_pairs, cur_data_pairs, bias_pairs = [], [], []
    trajectory_bias_list = []

    base_name = os.path.split(data_args.cache_file)[-1].replace('.json', '') + f'_{data_args.sample_ratios}_{data_args.sample_bins}'
    for idx in tqdm(range(len(train_dataset._samples))):
        cur_data = train_dataset._samples[idx]
        # distributed_dict['steering_control'].append(cur_data['steering_control'])
        # distributed_dict['throttle_and_brake_control'].append(cur_data['throttle_and_brake_control'])
        # distributed_dict['current_steering'].append(cur_data['current_steering'])
        # distributed_dict['current_throttle'].append(cur_data['current_throttle'])
        trajectory_bias_list.append(cur_data['trajectory_bias'])
        # data_pairs.append((cur_data['steering_control'], cur_data['throttle_and_brake_control']))
        # cur_data_pairs.append((cur_data['current_steering'], cur_data['current_throttle']))
        # bias_pairs.append((
        #     'steering_bias' if cur_data['steering_bias'] else 'steering_aligned',
        #     'throttle_bias' if cur_data['throttle_bias'] else 'throttle_aligned',
        # ))

    # plot_and_save_distributions(distributed_dict, prefix=base_name)
    # plot_and_save_heatmap(data_pairs, steering_classes=['Go Straight', 'Slight Left', 'Moderate Left', 'Turn Left', 'Slight Right', 'Moderate Right', 'Turn Right'], longitudinal_classes=['Hard Accelerate', 'Middle Accelerate', 'Light Accelerate', 'Braking', 'Coasting'], file_name=f'{base_name}_steering_longitudinal')
    # plot_and_save_heatmap(cur_data_pairs, steering_classes=['Go Straight', 'Slight Left', 'Moderate Left', 'Turn Left', 'Slight Right', 'Moderate Right', 'Turn Right'], longitudinal_classes=['Hard Accelerate', 'Middle Accelerate', 'Light Accelerate', 'Braking', 'Coasting'], file_name=f'{base_name}_steering_longitudinal_current')
    # plot_and_save_heatmap(bias_pairs, steering_classes=['steering_bias', 'steering_aligned'], longitudinal_classes=['throttle_bias', 'throttle_aligned'], file_name=f'{base_name}_bias')
    plot_float_distribution(trajectory_bias_list, global_min=0, global_max=50, save_filename=f'{base_name}_trajectory_bias')
    plot_float_distribution(trajectory_bias_list, global_min=50, global_max=150, save_filename=f'{base_name}_trajectory_bias')
    plot_float_distribution(trajectory_bias_list, global_min=150, global_max=1000, save_filename=f'{base_name}_trajectory_bias')


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, NuScenesDataConfig, SFTConfig, ModelConfig))
    script_args, data_args, sft_args, model_args = parser.parse_args_and_config()
    main(script_args, data_args, sft_args, model_args)
