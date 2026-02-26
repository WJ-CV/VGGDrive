import torch
from vggt.models.vggt import VGGT
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import autocast

class VGGTWrapper:
    _instance = None
    
    @classmethod
    def get_instance(cls, device='cpu'):
        if cls._instance is None:
            cls._instance = VGGT()
            vggt_state_dict = torch.load('vggt/model.pt', map_location='cpu')
            cls._instance.load_state_dict(vggt_state_dict)
            cls._instance.eval()
            for param in cls._instance.parameters():
                param.requires_grad = False
            cls._instance._is_vggt_inference = True
        return cls._instance.to(device)

def run_vggt_inference(images, device='cuda'):
    vggt_model = VGGTWrapper.get_instance(device)
    
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        aggregated_tokens_list, ps_idx = vggt_model.aggregator(images)

    final_token = aggregated_tokens_list[-1].detach()
    final_token = final_token.to(device).clone()
    return final_token

class UpsampleWithPixelUnshuffle(nn.Module):
    def __init__(self, in_channels=3584, factor_1=4, factor_2=7, out_channels=96):
        super().__init__()
        self.factor_1 = factor_1
        self.factor_2 = factor_2
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear = nn.Linear(self.in_channels // (self.factor_1 ** 2), self.out_channels * (self.factor_2 ** 2))
        self.linear_out = nn.Linear(self.out_channels, 1)

    def forward(self, x):
        x_1 = F.pixel_shuffle(x, upscale_factor=self.factor_1)
        B, C, H, W = x_1.shape

        x_trans = x_1.flatten(2, 3).permute(0, 2, 1) # 

        x_2 = self.linear(x_trans).permute(0, 2, 1)
        x_2_trans = x_2.view(B, -1, H, W)

        x_3 = F.pixel_shuffle(x_2_trans, upscale_factor=self.factor_2)
        x_3 = x_3.flatten(2, 3).permute(0, 2, 1) # 

        x_out = self.linear_out(x_3).permute(0, 2, 1)
        x_out = x_out.view(B, 1, H * self.factor_2, W * self.factor_2)

        return x_out