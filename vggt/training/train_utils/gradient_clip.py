import torch
import torch.nn as nn
from typing import Union, Optional




class GradientClipper:
    def __init__(self, configs, *args, **kwargs):
        """
        Args:
            configs: List of dictionaries, each containing:
                - module_name: str or list of str, module names to apply clipping to
                - max_norm: float, maximum norm for gradient clipping
                - norm_type: int, type of norm (default: 2)
        """
        self.configs = []
        self._first_call = True
        for config in configs:
            module_names = config['module_name']
            if isinstance(module_names, str):
                module_names = [module_names]

            self.configs.append({
                'module_names': module_names,
                'max_norm': float(config['max_norm']) if config['max_norm'] is not None else None,
                'norm_type': config.get('norm_type', 2)
            })


    def __call__(self, model: nn.Module) -> Optional[torch.Tensor]:
        # TODO: fix this, only record the parameter names once
        import pdb;pdb.set_trace()


        if self.max_norm is None:
            return None  # no-op

        # First, collect all parameters that should be clipped based on module names
        params_to_clip_by_config = []
        if self._first_call:
            all_clipped_params = set()

        for config in self.configs:
            current_config_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    for module_name in config['module_names']:
                        if module_name in name:
                            current_config_params.append(param)
                            if self._first_call:
                                all_clipped_params.add(param)
                            break
            params_to_clip_by_config.append((config, current_config_params))

        # Check for remaining parameters only on first call
        if self._first_call:
            remaining_params = []
            for name, param in model.named_parameters():
                if param.requires_grad and param not in all_clipped_params:
                    remaining_params.append(param)

            if len(remaining_params) > 0:
                print(f"Found {len(remaining_params)} parameters that won't be clipped")
                print(remaining_params)
                raise ValueError("Some parameters are not configured for gradient clipping")
            self._first_call = False

        grad_norms = {}
        for config, params_to_clip in params_to_clip_by_config:
            if not params_to_clip or config['max_norm'] is None:
                continue

            grad_norm = nn.utils.clip_grad_norm_(
                params_to_clip,
                max_norm=config['max_norm'],
                norm_type=config['norm_type']
            )

            if grad_norm is None:
                continue
            grad_norms[config] = grad_norm
        return grad_norms
