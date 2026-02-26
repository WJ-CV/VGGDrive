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
from depth_utils.vggt_utils import run_vggt_inference
@dataclass
class Qwen2_5_VLCausalLMOutputWithPast_(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.FloatTensor] = None
    dist_loss: Optional[torch.FloatTensor] = None

def Hidden_states_trans_inv_func(vision_hidden_states, hidden_states, img_mask):
    B = vision_hidden_states.shape[0]

    hidden_states_out = hidden_states.clone()
    for b in range(B):
        batch_mask = img_mask[b]  # [token_count]
        hidden_states_out[b][batch_mask] = vision_hidden_states[b]

    return hidden_states_out

def Hidden_states_trans_func(hidden_states, vggt_visual_gt, img_mask):
    vggt_visual_gt = vggt_visual_gt.contiguous()
    B, S, _, in_dim = vggt_visual_gt.shape

    vggt_visual_gt = vggt_visual_gt.view(B, -1, in_dim)
    vggt_gt_final = vggt_visual_gt.to(dtype=hidden_states.dtype, device=hidden_states.device)

    vision_hidden_states_list = []
    for b in range(B):
        batch_mask = img_mask[b]
        batch_vision_feat = hidden_states[b][batch_mask]
        vision_hidden_states_list.append(batch_vision_feat)
    
    vision_hidden_states = torch.stack(vision_hidden_states_list, dim=0)
    return vision_hidden_states, vggt_gt_final

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, F1, F2):
        """
        F1: [B, N, 512] - Query (from LLM)
        F2: [B, N, 512] - Key/Value (from 3D model)
        Returns:
            [B, N, 512] - Attention-enhanced feature
        """
        B, N, _ = F1.shape
        N_2 = F2.shape[1]

        Q = self.q_proj(F1).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) 
        K = self.k_proj(F2).view(B, N_2, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(F2).view(B, N_2, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # [B, h, N, d]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, -1)

        out = self.out_proj(attn_output)
        return self.layernorm(F1 + out) 
        
class CustomQwen2_5_VLModel(Qwen2_5_VLModel):
    def __init__(self, config):
        super().__init__(config)
        self.HS = getattr(config, "hidden_size", 3584)
        self.num_layers = getattr(config, "num_hidden_layers", 28)
        self.in_dim = 2048
        self.scale = 4

        self.prompt_tuning_mlp = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.HS, self.in_dim // self.scale),
                    nn.GELU(),
                    nn.Linear(self.in_dim // self.scale, self.in_dim // self.scale),
                ),
                nn.Sequential(
                    nn.Linear(self.in_dim, self.in_dim // self.scale),
                    nn.GELU(),
                    nn.Linear(self.in_dim // self.scale, self.in_dim // self.scale),
                ),
                CrossAttentionFusion(dim=self.in_dim // self.scale),
                nn.Sequential(
                    nn.Linear(self.in_dim // self.scale, self.in_dim // self.scale),
                    nn.GELU(),
                    nn.Linear(self.in_dim // self.scale, self.HS),
                )
            ])
            for _ in range(self.num_layers)
        ])
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        feature_3D: Optional[torch.FloatTensor] = None,
        img_mask: Optional[torch.BoolTensor] = None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        if self.training:
            use_3D_inject = True
        elif len(past_key_values.key_cache) == 0:
            use_3D_inject = True
        else:
            use_3D_inject = False

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            ########## prompt_tuning & output all hidden_states  #############
            if use_3D_inject:
                B, S, tokens, _ = feature_3D.shape
                vision_hidden_states, vggt_3D_feat = Hidden_states_trans_func(hidden_states, feature_3D, img_mask)
                vision_hs = self.prompt_tuning_mlp[idx][0](vision_hidden_states)
                vggt_hs = self.prompt_tuning_mlp[idx][1](vggt_3D_feat) 

                ## cross_attention (vision feat & vggt 3D feat)
                enhanced_vision_hs = self.prompt_tuning_mlp[idx][2](vision_hs, vggt_hs)

                # Residual
                enhanced_vision_hs = vision_hidden_states + self.prompt_tuning_mlp[idx][3](enhanced_vision_hs)
                hidden_states = Hidden_states_trans_inv_func(enhanced_vision_hs, hidden_states, img_mask)
            ####################

            if use_cache: 
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions: 
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class CustomQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.resized_W = kwargs.get("resized_W", 896)
        self.resized_H = kwargs.get("resized_H", 448)
        self.num_view = kwargs.get("num_views", 6)
        self.hidden_size = getattr(config, "hidden_size", 3584)

        self.model = CustomQwen2_5_VLModel(config)

    def generate(self, *args, **kwargs):
        self._img_list = kwargs.pop("img_list", None)
        return super().generate(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        img_list=None,
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
        if img_list is None:
            img_list = getattr(self, "_img_list", None)
        imgs_tensor = torch.stack(img_list, dim=0)
  
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions 
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states 
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:   
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw) 

                n_image_tokens = (input_ids == self.config.image_token_id).sum().item() 
                n_image_features = image_embeds.shape[0] 
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds) 
                image_mask = mask_expanded.to(inputs_embeds.device) 

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            
                vggt_visual = run_vggt_inference(
                    imgs_tensor.to(inputs_embeds.device),
                    device=inputs_embeds.device
                )
            else:
                vggt_visual = None
                mask = None

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
            feature_3D=vggt_visual,
            img_mask=mask,
        )

        last_hidden_states = outputs[0]
        logits = self.lm_head(last_hidden_states)
        loss = None
        dist_align_loss = None
        
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
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