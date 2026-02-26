# coding=utf-8
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

from dataclasses import dataclass, field
from typing import Optional

import trl


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None, metadata={"help": "The optional system prompt to use for benchmarking."}
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """
    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    notes: str = field(
        default="",
        metadata={'help': 'notes'}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )

@dataclass
class NuScenesDataConfig:
    """
    Arguments for specifying data input for training and evaluation.
    """
    version: str = field(
        default='v1.0-trainval',
        metadata={'help': 'dataset version'}
    )
    meta_dir: str = field(
        default="data/symlink_outputs_meta/nuscenes/",
        metadata={'help': 'directory for meta information of the dataset'}
    )
    split: str = field(
        default="mini_train",
        metadata={'help': 'which data split to use (e.g., train, val, test)'}
    )
    query_prompt_type: str = field(
        default="json",
        metadata={'help': 'which training prompt type'}
    )
    cache_file: str = field(
        default=""
    )
    sample_size: str = field(
        default="-1"
    )
    sample_bins: str = field(
        default=""
    )
    sample_ratios: str = field(
        default=""
    )
    obs_len: int = field(
        default=4,
        metadata={'help': 'observation length in timesteps'}
    )
    fut_len: int = field(
        default=6,
        metadata={'help': 'future prediction length in timesteps'}
    )
    num_views: int = field(
        default=3,
        metadata={"help": "Number of camera views (e.g., front-left, front, front-right)"}
    )
    img_weight: int = field(
        default=1920,
        metadata={"help": "Original image width"}
    )
    img_height: int = field(
        default=1080,
        metadata={"help": "Original image height"}
    )
    re_weight: int = field(
        default=896,
        metadata={"help": "Resized image width"}
    )
    re_height: int = field(
        default=448,
        metadata={"help": "Resized image height"}
    )
    rebuild: bool = field(
        default=True,
        metadata={'help': 'whether to rebuild the dataset cache'}
    )
    use_target: bool = field(
        default=False,
        metadata={'help': 'whether to include target vehicle information'}
    )
    use_action: bool = field(
        default=True,
        metadata={'help': 'whether to include action predictions'}
    )
    use_traj: bool = field(
        default=True,
        metadata={'help': 'whether to include trajectory predictions'}
    )
    use_nav: bool = field(
        default=False,
        metadata={'help': 'whether to include navigation information'}
    )
    num_path_points: int = field(
        default=10,
        metadata={'help': 'number of path points for trajectories'}
    )
    path_point_distance: float = field(
        default=3.0,
        metadata={'help': 'distance between path points'}
    )
    decimal: int = field(
        default=2,
        metadata={'help': 'number of decimal places to round coordinates'}
    )
    sft_indices: str = field(
        default="",
        metadata={'help': 'directory for sft data indices'}
    )
    balanced_action: str = field(
        default=""
    )
    append_datasets: str = field(
        default=""
    )
    mixed_shuffle: bool = field(
        default=False
    )
    force_load: bool = field(
        default=False
    )
    frame_start_interval: int = field(
        default=30
    )
    ensure_no_overlap: bool = field(
        default=True
    )
    ratio_shuffle: bool = field(
        default=False
    )
    required_cams: str = field(
        default=""
    )
    image_description: str = field(
        default=""
    )
    image_root: str = field(
        default=""
    )
    dataset_classname: str = field(
        default=None,
        metadata={"help": "The name of DatasetClass"}
    )
    omnidrive_vqa_folder: str = field(
        default="data/datasets/OmniDrive/data_nusc",
        metadata={'help': 'The path of omnidrive'}
    )
    use_prompt_mask: bool = field(
        default=True,
        metadata={'help': 'whether maskout the prompt labels or not'}
    )
    aligned_sample_ratio: float = field(
        default=-1
    )
    aligned_steering_sample_ratio: float = field(
        default=-1
    )
    aligned_throttle_sample_ratio: float = field(
        default=-1
    )


def preprocess_data_args(args):
    args.sample_size = [int(i) for i in args.sample_size.split('|')]
    args.sample_ratios = args.sample_ratios.split('|')
    args.sample_bins = args.sample_bins.split('|')
    args.cache_file = args.cache_file.split('|')
    return args

def index_data_args(args, i):
    args.sample_size = args.sample_size[i]
    args.sample_ratios = args.sample_ratios[i]
    args.sample_bins = args.sample_bins[i]
    args.cache_file = args.cache_file[i]
    return args
