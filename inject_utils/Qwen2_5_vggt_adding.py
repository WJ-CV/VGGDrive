import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionFlashAttention2,
    apply_rotary_pos_emb_flashatt,
    flash_attn_varlen_func,
    Qwen2_5_VLCausalLMOutputWithPast
)
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import List, Optional, Tuple, Union, Dict, Any
from inject_utils.vggt_utils import run_vggt_inference

@dataclass
@dataclass
class Qwen2_5_VLCausalLMOutputWithPast_(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.FloatTensor] = None
    dist_loss: Optional[torch.FloatTensor] = None

class VGGT_feat_adding_fusion(torch.nn.Module):
    def __init__(
        self,
        out_tokens_dim=3584,
        in_dim=2048,
        H=448,
        W=896,
        vggt_w=518,
        patch_size=14
        ):
        super(VGGT_feat_adding_fusion, self).__init__()
        self.out_tokens_dim = out_tokens_dim
        self.in_dim = in_dim

        self.pool_H = H // (patch_size * 2)  # 448 / 28 = 16
        self.pool_W = W // (patch_size * 2)  # 896 / 28 = 32
        self.vggt_w = vggt_w // patch_size  # 518 / 14 = 37

        self.adapt_avgpool = nn.AdaptiveAvgPool2d((self.pool_H, self.pool_W))
        self.dim_linear = nn.Linear(self.in_dim, self.out_tokens_dim)

    def forward(self, vision_hidden_states, vggt_visual_gt): # [4, 1536, 3584]  [B, S, 782, 2048]
        vggt_visual_gt = vggt_visual_gt[:, :, 5:, :] # [B, S, 777, 2048]
        B, S, _, _ = vggt_visual_gt.shape
        vision_hidden_states = vision_hidden_states.view(B, -1, self.out_tokens_dim)

        if vision_hidden_states.shape[1] == vggt_visual_gt.shape[1] * vggt_visual_gt.shape[2]:
            vggt_gt_final = vggt_visual_gt.reshape(B, -1, self.in_dim)
            vggt_gt_final = vggt_gt_final.to(dtype=vision_hidden_states.dtype, device=vision_hidden_states.device)
            vggt_visual_feat = self.dim_linear(vggt_gt_final) 
        else:
            vggt_visual_gt = vggt_visual_gt.view(B, S, -1, self.vggt_w, self.in_dim) # [B, S, 21, 37, 2048]
            vggt_visual_gt = vggt_visual_gt.flatten(0,1).permute(0, 3, 1, 2) #[B*S, 2048, 21, 37]
            vggt_gt_pooled = self.adapt_avgpool(vggt_visual_gt) #[B*S, 2048, 16, 32]
            vggt_gt_final = vggt_gt_pooled.view(B*S, self.in_dim, -1).permute(0, 2, 1) #[B*S, 512, 2048]

            vggt_gt_final = vggt_gt_final.to(dtype=vision_hidden_states.dtype, device=vision_hidden_states.device)

            vggt_visual_feat = self.dim_linear(vggt_gt_final) # # [B*S, 512, 3584]
            vggt_visual_feat = vggt_visual_feat.view(B, -1, self.out_tokens_dim)

        fusion_visual_feat = vision_hidden_states + vggt_visual_feat # [B, S*512, 3584]

        return fusion_visual_feat

class CustomQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.resized_W = kwargs.get("resized_W", 896)
        self.resized_H = kwargs.get("resized_H", 448)
        self.num_view = kwargs.get("num_views", 6)
        self.hidden_size = getattr(config, "hidden_size", 3584)

        self.feat_fusion = VGGT_feat_adding_fusion(H=self.resized_H, W=self.resized_W)

    def generate(self, *args, **kwargs):
        self._img_list = kwargs.pop("img_list", None)
        self._cam2lidar = kwargs.pop("cam2lidar", None)
        return super().generate(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        img_list=None,
        cam2lidar=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        pixel_values_videos=None,
    ):
 
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  # False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states  # False
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict # True

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids) # [B, 2219] -> [B, 2219, 3584]
            if pixel_values is not None: # [16384, 1176]   [B*4*32*64, 2*14*14]   
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw) # [4096, 3584]

                n_image_tokens = (input_ids == self.config.image_token_id).sum().item() # 4096 统计 image token 个数
                n_image_features = image_embeds.shape[0] # 图像特征个数
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id # [B, 2199] 标记哪些位置是 image token
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds) # [2, 2219, 3584]
                image_mask = mask_expanded.to(inputs_embeds.device) # [2, 2219, 3584]

                if img_list is None:
                    img_list = getattr(self, "_img_list", None)
                imgs_tensor = torch.stack(img_list, dim=0)

                vggt_visual = run_vggt_inference(
                    imgs_tensor.to(inputs_embeds.device),
                    device=inputs_embeds.device
                )
                image_embeds_fusion = self.feat_fusion(image_embeds.clone(), vggt_visual) # [4, 1536, 3584]  [B, S, 782, 2048] image_embeds.clone()

                image_embeds_fusion = image_embeds_fusion.to(inputs_embeds.device, inputs_embeds.dtype)# [4096, 3584]
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_fusion)  # [2, 2219, 3584]

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        last_hidden_states = outputs[0] # [2, 2219, 3584]
        logits = self.lm_head(last_hidden_states) # [2, 2219, 152064]
        loss = None
        dist_align_loss = None
        
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous() # [2, 2198, 152064]
            shift_labels = labels[..., 1:].contiguous() # [2, 2199] --> [2, 2198]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size) # [b*2198, 152064]
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast_(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            dist_loss=dist_align_loss,
        )