from typing import Tuple, List, Optional, Dict
import logging
from pathlib import Path
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils import comm
import open_clip
from open_clip.modified_resnet import ModifiedResNet, AttentionPool2d

from detectron2.config import configurable
from detectron2.modeling import ShapeSpec

from .build import VISUAL_ENCODER_REGISTRY
from .base import BaseVisualEncoder, PaddedList
from ..utils.misc import downsample_masks
from ..utils.ckpt import load_state_dict_with_beg_key


_logger = logging.getLogger(__name__)


@VISUAL_ENCODER_REGISTRY.register()
class CLIPResNet(BaseVisualEncoder):
    @configurable
    def __init__(self, cfg):
        super().__init__(cfg)
        model_name = cfg.MODEL_NAME
        pretrained = cfg.PRETRAINED
        self._out_features = cfg.OUT_FEATURES
        
        assert 'rn' in model_name.lower()
        if comm.get_local_rank() == 0:
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        comm.synchronize()
        clip_model = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)[0]
        self.proxy: ModifiedResNet = clip_model.visual
        
        if len(cfg.LOAD_FROM) > 0:
            load_from_path = Path(cfg.LOAD_FROM)
            if not load_from_path.exists():
                raise FileNotFoundError(f"LOAD_FROM {load_from_path} does not exist")
            load_ckpt = torch.load(load_from_path, map_location='cpu')
            if 'model' in load_ckpt:
                load_ckpt = load_ckpt['model']
            load_state_dict_with_beg_key(self.proxy, load_ckpt, cfg.LOAD_BEG_KEY, 
                                         self.__class__.__name__+".proxy", load_from_path, _logger)
            
        self._output_strides = {}
        self._output_channels = {}
        
        for i in range(1, 5):
            layer = getattr(self.proxy, f"layer{i}")
            self._output_channels[f"layer{i}{self._feature_suffix}"] = layer[-1].bn3.weight.shape[0]
            self._output_strides[f"layer{i}{self._feature_suffix}"] = 2 ** (i+1)
        
        embed_dim = clip_model.text_projection.shape[-1]
        self._output_channels[f"emb{self._feature_suffix}"] = embed_dim
        self._output_strides[f"emb{self._feature_suffix}"] = -1
        self._output_channels[f"m_embs{self._feature_suffix}"] = embed_dim
        self._output_strides[f"m_embs{self._feature_suffix}"] = -1
        self._output_channels[f"i_embs{self._feature_suffix}"] = embed_dim
        self._output_strides[f"i_embs{self._feature_suffix}"] = -1
        self._output_channels[f"p_embs{self._feature_suffix}"] = embed_dim
        self._output_strides[f"p_embs{self._feature_suffix}"] = 32
        
        self._pretrained_grid_size = (self.proxy.image_size // 32, self.proxy.image_size // 32)
        
        self._downsample_method = cfg.DOWNSAMPLE_METHOD
        self._down_mask_thresh = cfg.DOWN_MASK_THRESH
        self._mask_prior_beg = cfg.MASK_PRIOR_BEG  # no use
        self.neck_heads = self.proxy.attnpool.num_heads
        init_mask_logit_scale = torch.full((self.neck_heads, ), cfg.MASK_LOGIT_SCALE)
        if cfg.LEARNABLE_MASK_LOGIT_SCALE:
            self.register_parameter("mask_logit_scale", nn.Parameter(init_mask_logit_scale))
        else:
            self.register_buffer("mask_logit_scale", init_mask_logit_scale, persistent=False)
        
        self._freeze(cfg.FINETUNE_TYPE)
        
    @classmethod
    def from_config(cls, cfg, all_cfg):
        return {"cfg": cfg}
    
    def _freeze(self, finetune_type):
        if finetune_type == 'all':
            return
        if finetune_type == 'all_but_bn':
            freeze_names = []
            for name, param in self.proxy.named_parameters():
                if 'bn' in name:
                    param.requires_grad = False
                    freeze_names.append(name)
                else:
                    param.requires_grad = True
            _logger.info(f"Freeze {len(freeze_names)} parameters in {self.__class__.__name__}: {freeze_names}")
        elif finetune_type == 'none':
            for p in self.parameters():
                p.requires_grad = False
            return
        else:
            raise ValueError(f"Unknown finetune type {finetune_type} for {self.__class__.__name__}")
    
    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        return {
            name: ShapeSpec(
                channels=self._output_channels[name], stride=self._output_strides[name]
            )
            for name in self._output_channels.keys()
        }
        
    def extract_features(self, inputs: PaddedList) -> Dict[str, Tensor]:
        if self._finetune_none:
            self.eval()
            with torch.no_grad():
                return self._extract_features(inputs.images, inputs.masks)
        else:
            return self._extract_features(inputs.images, inputs.masks)
        
    def _extract_features(self, x: Tensor, masks: List[Tensor]):
        out = {}
        # stem
        x = self.proxy.act1(self.proxy.bn1(self.proxy.conv1(x)))
        x = self.proxy.act2(self.proxy.bn2(self.proxy.conv2(x)))
        x = self.proxy.act3(self.proxy.bn3(self.proxy.conv3(x)))
        x = self.proxy.avgpool(x)
        x = self.proxy.layer1(x)
        if f"layer1{self._feature_suffix}" in self._out_features:
            out[f"layer1{self._feature_suffix}"] = x
        x = self.proxy.layer2(x)
        if f"layer2{self._feature_suffix}" in self._out_features:
            out[f"layer2{self._feature_suffix}"] = x
        x = self.proxy.layer3(x)
        if f"layer3{self._feature_suffix}" in self._out_features:
            out[f"layer3{self._feature_suffix}"] = x
        x = self.proxy.layer4(x)
        if f"layer4{self._feature_suffix}" in self._out_features:
            out[f"layer4{self._feature_suffix}"] = x
        pk = f"p_embs{self._feature_suffix}"      
        mk = f"m_embs{self._feature_suffix}" 
        ik = f"i_embs{self._feature_suffix}"
        return_p_embs = pk in self._out_features
        return_i_embs = ik in self._out_features

        m_embs, areas, p_embs, i_embs = self._attention_pool_with_masks(self.proxy.attnpool, x, masks, 
                                                         return_p_embs=return_p_embs,
                                                         return_i_embs=return_i_embs)
        out[f"areas{self._feature_suffix}"] = areas
        if return_p_embs:
            out[pk] = p_embs
        if return_i_embs:
            out[ik] = i_embs[:, :, None, None]
        if mk in self._out_features:
            out[mk] = m_embs
        return out
    
    def _interpolate_postion_embedding(self, curr_grid_size: Tuple[int, int]) -> Tensor:
        pretrained_grid_size = self._pretrained_grid_size
        pretrained_pe = self.proxy.attnpool.positional_embedding
        if curr_grid_size == pretrained_grid_size:
            return pretrained_pe
        global_pe, local_pe = pretrained_pe[0:1], pretrained_pe[1:]
        local_pe = local_pe = local_pe.reshape(pretrained_grid_size[0], pretrained_grid_size[1], local_pe.shape[-1]).permute(2, 0, 1).unsqueeze(0)
        interp_local_pe = F.interpolate(local_pe, size=curr_grid_size, mode='bicubic', align_corners=True)
        interp_local_pe = interp_local_pe.squeeze(0).permute(1, 2, 0).reshape(curr_grid_size[0]*curr_grid_size[1], -1)
        interp_pe = torch.cat([global_pe, interp_local_pe], dim=0)
        return interp_pe
    
    def _build_attn_mask(self, masks, curr_gird_size):
        num_masks = [mask.shape[0] for mask in masks]
        B = len(masks)
        Q = max(num_masks)
        H, W = curr_gird_size
        L = H * W
        minimum = -100
        nh = self.mask_logit_scale.shape[0]
        attn_mask = torch.full((B, nh, L+Q, L), minimum, dtype=torch.float32)
        attn_mask[:, :, torch.arange(L), torch.arange(L)] = 0
        areas = []
        for i, (mask, num_mask) in enumerate(zip(masks, num_masks)):
            if num_mask == 0:
                areas.append(torch.zeros(0, dtype=mask.dtype, device=mask.device))
                continue
            down_mask = downsample_masks(mask, (H, W), method=self._downsample_method)
            areas.append(down_mask.sum(dim=(1, 2)))
            if self._down_mask_thresh is None:
                th = down_mask.flatten(1).max(dim=1).values / 2
                th = th[:, None, None]
            else:
                th = self._down_mask_thresh
            bin_down_mask = down_mask > th
            down_mask = down_mask[None, :, :, :] * self.mask_logit_scale.exp()[:, None, None, None]  # nh, num_mask, H, W
            down_mask[:, ~bin_down_mask] = minimum
            attn_mask[i, :, L:L+num_mask] = down_mask.flatten(2)
        attn_mask = attn_mask.view(B*nh, L+Q, L)
        return attn_mask, areas
    
    def _attention_pool_with_masks(self,
                                   attnpool: AttentionPool2d,
                                   x: Tensor, 
                                   masks: List[Tensor],
                                   return_p_embs: bool,
                                   return_i_embs: bool) -> Tuple[List[Tensor], List[Tensor], Optional[Tensor], Optional[Tensor]]:
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(2, 0, 1)   # (HW)BC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0) 
        pe = self._interpolate_postion_embedding((H, W))
        x = x + pe[:, None, :].to(x.dtype)
        if return_i_embs:
            i_embs, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=attnpool.num_heads,
                q_proj_weight=attnpool.q_proj.weight,
                k_proj_weight=attnpool.k_proj.weight,
                v_proj_weight=attnpool.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([attnpool.q_proj.bias, attnpool.k_proj.bias, attnpool.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0.,
                out_proj_weight=attnpool.c_proj.weight,
                out_proj_bias=attnpool.c_proj.bias,
                use_separate_proj_weight=True,
                training=attnpool.training,
                need_weights=False
            )
            i_embs = i_embs[0]
        else:
            i_embs = None
        attn_mask, areas = self._build_attn_mask(masks, (H, W))
        attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)
        Q = attn_mask.shape[1] - H * W
        L = H * W
        if return_p_embs:
            query = torch.cat([x[1:], x[0:1].repeat(Q, 1, 1)], dim=0)
        else:
            query = x[0:1].repeat(Q, 1, 1)
            attn_mask = attn_mask[:, L:, :]  # B,Q,L
        key_value = x[1:]
        query, _ = F.multi_head_attention_forward(
            query=query, key=key_value, value=key_value,
            embed_dim_to_check=x.shape[-1],
            num_heads=attnpool.num_heads,
            q_proj_weight=attnpool.q_proj.weight,
            k_proj_weight=attnpool.k_proj.weight,
            v_proj_weight=attnpool.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([attnpool.q_proj.bias, attnpool.k_proj.bias, attnpool.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=attnpool.c_proj.weight,
            out_proj_bias=attnpool.c_proj.bias,
            use_separate_proj_weight=True,
            training=attnpool.training,
            need_weights=False,
            attn_mask=attn_mask
        )
        if return_p_embs:
            p_embs = query[:L,]  # (HW)BD
            p_embs = p_embs.permute(1, 2, 0).reshape(B, -1, H, W).contiguous()
        else:
            p_embs = None

        padded_m_embs = query[-Q:,]  # (Q)BD
        padded_m_embs = padded_m_embs.transpose(0, 1).contiguous()
        m_embs = []
        for i, mask in enumerate(masks):
            num_mask = mask.shape[0]
            m_embs.append(padded_m_embs[i, :num_mask])
        return m_embs, areas, p_embs, i_embs