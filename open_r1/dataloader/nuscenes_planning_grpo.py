from open_r1.dataloader.nuscenes_planning import NuScenesPlanningSupervisedDataset

import os
from open_r1.configs import NuScenesDataConfig
from open_r1.dataloader.utils import preprocess_meta_action_str, QUERY_PROMPT
from PIL import Image

from qwen_vl_utils import process_vision_info

class NuScenesPlanningGRPODataset(NuScenesPlanningSupervisedDataset):
    def __init__(
        self,
        data_path: str,
        data_args: NuScenesDataConfig
    ):
        super(NuScenesPlanningGRPODataset, self).__init__(data_path=data_path, data_args=data_args)

    def __getitem__(self, idx):
        info = self._samples[idx]

        if 'speed' in self.query_prompt_type:
            prompt = QUERY_PROMPT[self.query_prompt_type].replace('__speed__', f'{info["speed"]:.2f}')
        else:
            prompt = QUERY_PROMPT[self.query_prompt_type]
        lat_action = info['lat_action']  # lateral control
        lon_action = info['lon_action']  # longitudinal control
        lat_action, lon_action = preprocess_meta_action_str(lat_action), preprocess_meta_action_str(lon_action)

        message = [
            {
                "role": "user",
                "content":
                    [{"type": "image", "image": f"file://{os.path.join(self.dataroot, image_path)}", **self.resize_kwargs} for image_path in info['frames']] + \
                    [{"type": "text", "text": prompt}],
            }
        ]
        image_inputs, video_inputs = process_vision_info(message)
        return {
            "message": message,
            "lat_action": lat_action,
            "lon_action": lon_action,
            "trajectory": info['fut_trajectory'].tolist(),
            'image_inputs': image_inputs,
            'video_inputs': video_inputs
        }
