# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional, Set, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig


def build_optimizer(model: nn.Module, config: DictConfig):
    """
    Main entry point for building optimizers.

    Expected config structure:
    ```
    optimizer:
      _target_: torch.optim.AdamW
      lr: 1e-4
      weight_decay: 0.05

    # Optional scheduling
    lr_schedule:
      _target_: some.scheduler.Class

    # Optional parameter grouping
    param_groups:
      lr:
        "*.bias": 2.0  # 2x learning rate for bias terms
      weight_decay:
        "*.bias": 0.0  # no weight decay for bias
    ```
    """
    return create_optimizer(
        model=model,
        optimizer_config=config.optimizer,
        lr_schedule=config.get("lr_schedule"),
        weight_decay_schedule=config.get("weight_decay_schedule"),
        param_group_config=config.get("param_groups"),
    )



class OptimizerWrapper:
    """Simple wrapper around PyTorch optimizers with scheduler support."""

    def __init__(self, optimizer, schedulers=None):
        self.optimizer = optimizer
        self.schedulers = schedulers or []
        self._validate_schedulers()
        self.step_schedulers(0.0)

    def _validate_schedulers(self):
        """Validate that scheduler options exist in optimizer defaults."""
        for scheduler_group in self.schedulers:
            for option in scheduler_group.keys():
                if option not in self.optimizer.defaults:
                    raise ValueError(
                        f"Optimizer option '{option}' not found. "
                        f"Valid options: {list(self.optimizer.defaults.keys())}"
                    )

    def step_schedulers(self, progress: float):
        """Update optimizer parameters based on training progress (0.0 to 1.0)."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i < len(self.schedulers):
                for option, scheduler in self.schedulers[i].items():
                    param_group[option] = scheduler(progress)

    def step(self, progress: float, closure=None):
        """Step the optimizer and update schedulers."""
        self.step_schedulers(progress)
        return self.optimizer.step(closure)

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)


def create_param_groups(
    model: nn.Module,
    lr_config: Optional[Dict] = None,
    weight_decay_config: Optional[Dict] = None,
) -> List[Dict]:
    """
    Create parameter groups with different learning rates and weight decay.

    Args:
        model: PyTorch model
        lr_config: Dict mapping parameter name patterns to learning rate multipliers
        weight_decay_config: Dict mapping parameter name patterns to weight decay values

    Returns:
        List of parameter group dictionaries
    """
    if not lr_config and not weight_decay_config:
        return [{"params": list(model.parameters())}]

    param_groups = []
    processed_params = set()

    # Get all named parameters
    named_params = dict(model.named_parameters())

    # Process custom configurations
    configs = {}
    if lr_config:
        configs.update({f"lr_{k}": v for k, v in lr_config.items()})
    if weight_decay_config:
        configs.update({f"wd_{k}": v for k, v in weight_decay_config.items()})

    # Group parameters by their configuration
    for config_name, config_value in configs.items():
        matching_params = []
        config_type = config_name.split('_')[0]  # 'lr' or 'wd'
        pattern = '_'.join(config_name.split('_')[1:])

        for param_name, param in named_params.items():
            if param in processed_params:
                continue
            if _matches_pattern(param_name, pattern):
                matching_params.append(param)
                processed_params.add(param)

        if matching_params:
            group = {"params": matching_params}
            if config_type == "lr":
                group["lr_multiplier"] = config_value
            elif config_type == "wd":
                group["weight_decay"] = config_value
            param_groups.append(group)

    # Add remaining parameters to default group
    remaining_params = [p for p in model.parameters() if p not in processed_params]
    if remaining_params:
        param_groups.append({"params": remaining_params})

    return param_groups


def _matches_pattern(param_name: str, pattern: str) -> bool:
    """Simple pattern matching for parameter names."""
    import fnmatch
    return fnmatch.fnmatch(param_name, pattern)


def create_optimizer(
    model: nn.Module,
    optimizer_config: DictConfig,
    lr_schedule: Optional[Any] = None,
    weight_decay_schedule: Optional[Any] = None,
    param_group_config: Optional[Dict] = None,
) -> OptimizerWrapper:
    """
    Create an optimizer with optional parameter grouping and scheduling.

    Args:
        model: PyTorch model to optimize
        optimizer_config: Hydra config for the optimizer (e.g., torch.optim.AdamW)
        lr_schedule: Learning rate scheduler function
        weight_decay_schedule: Weight decay scheduler function
        param_group_config: Configuration for different parameter groups

    Returns:
        OptimizerWrapper instance
    """
    # Create parameter groups
    if param_group_config:
        param_groups = create_param_groups(
            model,
            param_group_config.get("lr"),
            param_group_config.get("weight_decay")
        )
    else:
        param_groups = [{"params": list(model.parameters())}]

    # Instantiate optimizer
    try:
        import hydra
        optimizer = hydra.utils.instantiate(optimizer_config, param_groups)
    except ImportError:
        # Fallback without hydra
        optimizer_cls = optimizer_config._target_
        optimizer_kwargs = {k: v for k, v in optimizer_config.items() if k != "_target_"}
        optimizer = optimizer_cls(param_groups, **optimizer_kwargs)

    # Create schedulers
    schedulers = []
    for i, param_group in enumerate(param_groups):
        group_schedulers = {}
        if lr_schedule:
            group_schedulers["lr"] = lr_schedule
        if weight_decay_schedule:
            group_schedulers["weight_decay"] = weight_decay_schedule
        schedulers.append(group_schedulers)

    return OptimizerWrapper(optimizer, schedulers if any(schedulers) else None)
