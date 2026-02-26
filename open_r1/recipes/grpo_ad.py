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

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import random

from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from open_r1.trainer import Qwen2VLGRPOTrainer, GRPOConfig
from open_r1.configs import NuScenesDataConfig
from open_r1.dataloader.nuscenes_planning_grpo import NuScenesPlanningGRPODataset

import numpy as np

import json
answer_pattern = re.compile(r'<answer>([\s\S]*?)</answer>', re.IGNORECASE)

import logging
logger = logging.getLogger(__name__)

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


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


import numpy as np
import ast

def calculate_ade(p_traj, g_traj, s=3):
    distance = np.linalg.norm(p_traj[:s*2] - g_traj[:s*2], axis=1)
    return np.mean(distance)

def judge_trajectory_list(traj_list):
    if isinstance(traj_list, list) and len(traj_list) == 6:
        if all(isinstance(sublist, list) and len(sublist) == 2 for sublist in traj_list):
            if all(all(isinstance(num, (int, float)) for num in sublist) for sublist in traj_list):
                return True
    return False

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    think_pattern = r"<think>.*?</think>"
    lateral_control_pattern = r"<lateral_control>.*?</lateral_control>"
    longitudinal_control_pattern = r"<longitudinal_control>.*?</longitudinal_control>"
    trajectory_pattern = r"<trajectory>.*?</trajectory>"

    # 是否 < > 出现
    matches = [
        [re.findall(pattern, content, re.DOTALL) for content in completions] \
            for pattern in [think_pattern, lateral_control_pattern, longitudinal_control_pattern, trajectory_pattern]
    ]
    rewards = [
        [0.15 if match else 0.0 for match in i_matches] \
            for i_matches in matches
    ]
    rewards = [sum(items) for items in zip(*rewards)]
    all_in_rewards = [0 if i_rewards < 0.6 else 0.05 for i_rewards in rewards]

    # < > 的出现次数
    def count_to_reward(content, sub_str):
        score_1 = content.count(f'<{sub_str}>')
        score_2 = content.count(f'</{sub_str}>')
        if score_1 < 1 or score_2 < 1:
            return 0.0
        if score_1 > 1 or score_2 > 1:
            return -0.125

        return 0.0125

    count_rewards = [
        [count_to_reward(content, pattern_str) for content in completions] \
            for pattern_str in ['think', 'lateral_control', 'longitudinal_control', 'trajectory']
    ]
    count_rewards = [sum(items) for items in zip(*count_rewards)]

    trajectory_rewards, other_length_rewards = [], []
    for i_completion in completions:
        # 除了 < > 之外的文本
        other_text = i_completion
        for pattern in [think_pattern, lateral_control_pattern, longitudinal_control_pattern, trajectory_pattern]:
            other_text = re.sub(pattern, "", other_text, re.DOTALL).strip()
        if len(other_text) == 0:
            other_length_rewards.append(0.2)
        else:
            other_length_rewards.append(max(-0.1, 0.2 - 0.2 / 32 * len(other_text)))

        # trajectory 中间格式
        match = re.search(trajectory_pattern, i_completion, re.DOTALL)
        if not match:
            trajectory_rewards.append(0.0)
            continue
        traj_reward, traj_list = 0.0, []
        try:
            content = match.group()[len('<trajectory>'): -len('</trajectory>')].strip()
            traj_list = ast.literal_eval(content)
        except Exception as e:
            print('format_reward ast.literal_eval error', e, f'ecomp: {[i_completion]}')
        if traj_list is not None and isinstance(traj_list, list) and len(traj_list) > 0:
            try:
                if judge_trajectory_list(traj_list):
                    traj_reward = 0.5
                else:
                    traj_reward = -0.05
            except Exception as e:
                print('format_reward int, float error', e)
        else:
            traj_reward = -0.125

        trajectory_rewards.append(traj_reward)

    def judge_order(i_comp):
        think_pos = i_comp.find('</think>')
        if think_pos < i_comp.find('<lateral_control>') and \
            think_pos < i_comp.find('<longitudinal_control>') and \
                think_pos < i_comp.find('<trajectory>'):
                return True
        return False

    order_rewards = [
        0.1 if i_all_in > 0 and i_count >= 0.05 and i_other_length > 0 and judge_order(i_completion) else 0.0
        for i_all_in, i_count, i_other_length, i_completion in zip(all_in_rewards, count_rewards, other_length_rewards, completions)]

    final_rewards = [sum(items) for items in zip(rewards, trajectory_rewards, count_rewards, other_length_rewards, order_rewards)]
    if torch.cuda.current_device() == 0 and random.randint(1, 4) == 1:
        print('=' * 80)
        for i_completion in completions:
            print(torch.cuda.current_device(), 'e_completions', len(i_completion), [i_completion])
        print('e_rewards', rewards)
        print('e_trajectory_rewards', trajectory_rewards)
        print('e_count_rewards', count_rewards)
        print('e_other_length_rewards', other_length_rewards)
        print('e_order_rewards', order_rewards)
        print('e_final_rewards', final_rewards)
        print('=' * 80)
    return final_rewards

def extract_and_parse_token(text, token_left, token_right):
    pattern = token_left + r".*?" + token_right
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return False, ''
    try:
        content = match.group()[len(token_left): -len(token_right)].strip()
    except Exception as e:
        print(e)
        return False, ''
    return True, content

"""
/high_perf_store2/users/lg/users/zhangyikai/outputs/ad_r1/Qwen2.5-VL-3B-Instruct/03_25_08_45_35_270
"""

lateral_control_gt = ["Go Straight", "Turn Left", "Turn Right", "Deviate Left", "Deviate Right", "U Turn"]
longitudinal_control_gt = ["Maintain Speed", "Accelerate", "Decelerate", "Start from Zero", "Slow Down to Stop", "Hard Accelerate", "Hard Brake"]

def is_in_gt(cur_control, gt_list):
    return True if sum([cur_control.lower() in i_gt.lower() for i_gt in gt_list]) > 0 else False

def trajectory_reward(completions, lat_action, lon_action, trajectory, **kwargs):
    rewards = []
    for i_completion, i_lat_action, i_lon_action, i_trajectory in zip(
        completions, lat_action, lon_action, trajectory
    ):
        reward = 0.0
        preds = {}
        for k in ['lateral_control', 'longitudinal_control', 'trajectory']:
            preds[k] = ''
            try:
                is_extracted, preds[k] = extract_and_parse_token(i_completion, f'<{k}>', f'</{k}>')
            except Exception as e:
                print(f'[Error] {k}', e)

        # 是否在可选列表中：
        reward += (0.1 if len(preds['lateral_control']) > 0 and is_in_gt(preds['lateral_control'], lateral_control_gt) else -0.1)
        reward += (0.1 if len(preds['longitudinal_control']) > 0 and is_in_gt(preds['longitudinal_control'], lateral_control_gt) else -0.1)

        # meta action 是否正确
        if len(preds['lateral_control']) / len(i_lat_action) > 0.8 and preds['lateral_control'].lower() in i_lat_action.lower():
            reward += 0.2
        if len(preds['longitudinal_control']) / len(i_lon_action) > 0.8 and preds['longitudinal_control'].lower() in i_lon_action.lower():
            reward += 0.2

        # 回归 reward
        def trajectory_reward_func(ade, split_point):
            if ade < split_point:
                return 1 / (1 + ade)
            else:
                return 1 / (1 + split_point + 2 * (ade-split_point))
        traj_list = []
        if len(preds['trajectory']) > 0:
            try:
                traj_list = ast.literal_eval(preds['trajectory'])
            except Exception as e:
                print(f'[Error] trajectory ast.literal_eval', e)
        if judge_trajectory_list(traj_list):
            try:
                reward += trajectory_reward_func(calculate_ade(np.array(traj_list), np.array(i_trajectory), s=1), 0.4) * 2
                reward += trajectory_reward_func(calculate_ade(np.array(traj_list), np.array(i_trajectory), s=2), 1.6) * 3
                reward += trajectory_reward_func(calculate_ade(np.array(traj_list), np.array(i_trajectory), s=3), 2.4) * 3.66
            except Exception as e1:
                print('[Error] calculate_ade', e1)
        else:
            reward -= 0.4
        rewards.append(reward)
    if torch.cuda.current_device() == 0 and random.randint(1, 4) == 1:
        print('*' * 80)
        for i_completion in completions:
            print(torch.cuda.current_device(), 'e_completions', len(i_completion), [i_completion])
        print('e_t_rewards', rewards)
        print('e_t_lat_action', lat_action)
        print('e_t_lon_action', lon_action)
        print('e_t_trajectory', f'{trajectory}')
        print('*' * 80)
    return rewards

## json type rewards #########################################################################################

# def extract_and_parse_answer(text):
#     match = answer_pattern.search(text)
#     if not match:
#         return {"lateral_control": "", "longitudinal_control": "", "trajectory": []}

#     content = match.group(1).strip()

#     content = content.replace('```json', '').replace('```', '')
#     content = content.replace('\\"', '"').replace("\\'", "'").replace('\n', ' ')
#     content = re.sub(r"(\w+)\s*:", r'"\1":', content)  # 处理无引号key
#     content = re.sub(r"'", '"', content)  # 统一引号格式
#     content = re.sub(r',(\s*[]}])', r'\1', content)  # 移除尾部逗号

#     try:
#         parsed = json.loads(content, strict=False)
#     except Exception as e:
#         try:
#             content = re.sub(r'\[([\d\s.,]+)\]', 
#                            lambda m: '[' + ','.join([f'[{x}]' if not x.strip().startswith('[') else x for x in m.group(1).split(',')]) + ']', 
#                            content)
#             parsed = json.loads(content, strict=False)
#             if not isinstance(parsed, dict):
#                 parsed = {"lateral_control": "", "longitudinal_control": "", "trajectory": []}
#         except:
#             parsed = {"lateral_control": "", "longitudinal_control": "", "trajectory": []}

#     trajectory_list = []
#     for point in parsed.get("trajectory", []):
#         if not isinstance(point, (list, tuple)) or len(point) < 2:
#             continue
        
#         try:
#             x = float(point[0])
#             y = float(point[1])
#             trajectory_list.append([x, y])
#         except:
#             continue
    
#     return {
#         "lateral_control": str(parsed.get("lateral_control", "")),
#         "longitudinal_control": str(parsed.get("longitudinal_control", "")),
#         "trajectory": trajectory_list[:6]  # 保持最大6个点的限制
#     }

# def trajectory_reward(completions, lat_action, lon_action, trajectory, **kwargs):
#     rewards = []
#     for i_completion, i_lat_action, i_lon_action, i_trajectory in zip(
#         completions, lat_action, lon_action, trajectory):
#         reward, json_answer = 0.0, None
#         try:
#             preds = extract_and_parse_answer(i_completion)
#             if preds['lateral_control'].lower() in i_lat_action.lower() and len(preds['lateral_control']) / len(i_lat_action) > 0.8:
#                 reward += 1/2
#             if preds['longitudinal_control'].lower() in i_lon_action.lower() and len(preds['longitudinal_control']) / len(i_lon_action) > 0.8:
#                 reward += 1/2
#             if len(i_trajectory) == len(preds['trajectory']):
#                 try:
#                     reward += 1 / (1 + calculate_ade(np.array(preds['trajectory']), np.array(i_trajectory)))
#                 except Exception as e1:
#                     print('[Error] calculate_ade', e1)
#                     reward += 0.0
#         except Exception as e2:
#             print('[Error] trajectory_reward', e2)
#         rewards.append(reward)
#     return rewards

# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
#     matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completions]
#     return [1.0 if match else 0.0 for match in matches]

## json type rewards #########################################################################################


'''
    If the iou of the bbox predicted by the model and the ground truth is greater than 0.5, the reward is 1.0, otherwise 0.0 .
    This is a hard reward, maybe the soft reward is better and could be used in the future .
'''
def iou_reward(completions, solution, **kwargs):
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    # bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                bbox_match = re.search(bbox_pattern, content_answer)
                if bbox_match:
                    bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                    if iou(bbox, sol) > 0.5:
                        reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


reward_funcs_registry = {
    "accuracy": trajectory_reward,
    "format": format_reward
}

def main(script_args, data_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    # dataset = LazySupervisedDataset(script_args.dataset_name, script_args)
    dataset = NuScenesPlanningGRPODataset(script_args.dataset_name, data_args)

    trainer_cls = Qwen2VLGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint or None
    # logger.info(f"Resuming training from {checkpoint=}.")
    print(f"Resuming training from {checkpoint=}.")

    # Train and push the model to the Hub
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, NuScenesDataConfig, GRPOConfig, GRPOModelConfig))
    script_args, data_args, grpo_args, grpo_model_args = parser.parse_args_and_config()
    main(script_args, data_args, grpo_args, grpo_model_args)
