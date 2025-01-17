from typing import Tuple, List, Optional, Dict
import logging
from pathlib import Path
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils import comm
import open_clip
from open_clip.transformer import ResidualAttentionBlock

from detectron2.config import configurable
from detectron2.modeling import ShapeSpec

from .build import VISUAL_ENCODER_REGISTRY
from .base import BaseVisualEncoder, PaddedList
from ..utils.misc import downsample_masks
from ..utils.ckpt import load_state_dict_with_beg_key


_logger = logging.getLogger(__name__)

def _lnd2ndhw(x, grid_size):
    return x[1:].permute(1, 2, 0).reshape(x.shape[1], x.shape[-1], *grid_size).contiguous()

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


@VISUAL_ENCODER_REGISTRY.register()
class CLIPViT(BaseVisualEncoder):
    @configurable
    def __init__(self, cfg):
        super().__init__(cfg)
        model_name = cfg.MODEL_NAME
        pretrained = cfg.PRETRAINED
        
        # download on local rank 0 first
        if comm.get_local_rank() == 0:
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        comm.synchronize()
        
        clip_model = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)[0]
        
        self.proxy = clip_model.visual
        
        if len(cfg.LOAD_FROM) > 0:
            load_from_path = Path(cfg.LOAD_FROM)
            if not load_from_path.exists():
                raise FileNotFoundError(f"LOAD_FROM {load_from_path} does not exist")
            load_ckpt = torch.load(load_from_path, map_location='cpu')
            if 'model' in load_ckpt:
                load_ckpt = load_ckpt['model']
            load_state_dict_with_beg_key(self.proxy, load_ckpt, cfg.LOAD_BEG_KEY, 
                                         self.__class__.__name__+".proxy", load_from_path, _logger)
        
        self._out_features = cfg.OUT_FEATURES
        
        self._mask_prior_beg = cfg.MASK_PRIOR_BEG
        self._downsample_method = cfg.DOWNSAMPLE_METHOD
        self._down_mask_thresh = cfg.DOWN_MASK_THRESH
        
        self.neck_layers = len(self.proxy.transformer.resblocks) - self._mask_prior_beg + 1
        self.neck_heads = self.proxy.transformer.resblocks[-1].attn.num_heads
        init_mask_logit_scale = torch.full((self.neck_layers, self.neck_heads), cfg.MASK_LOGIT_SCALE)
        if cfg.LEARNABLE_MASK_LOGIT_SCALE:
            self.register_parameter("mask_logit_scale", nn.Parameter(init_mask_logit_scale))
        else:
            self.register_buffer("mask_logit_scale", init_mask_logit_scale, persistent=False)
        
        assert 0 < self._mask_prior_beg <= len(self.proxy.transformer.resblocks), f"Invalid mask prior begin index {self._mask_prior_beg}, should be in [1, {len(self.proxy.transformer.resblocks)}]"
        assert self.proxy.attn_pool is None
        assert self.proxy.pool_type == 'tok'
        self._output_strides = {}
        self._output_channels = {}
        for i in range(1, len(self.proxy.transformer.resblocks) + 1):
            self._output_strides[f"block{i}{self._feature_suffix}"] = self.proxy.patch_size[0]
            self._output_channels[f"block{i}{self._feature_suffix}"] = self.proxy.transformer.width
            if i == len(self.proxy.transformer.resblocks):
                mi = ''
            else:
                mi = f'-{len(self.proxy.transformer.resblocks) - i}'
            self._output_strides[f"m_embs{mi}{self._feature_suffix}"] = -1
            self._output_channels[f"m_embs{mi}{self._feature_suffix}"] = self.proxy.transformer.width
        self._output_strides[f"emb{self._feature_suffix}"] = -1
        self._output_channels[f"emb{self._feature_suffix}"] = clip_model.text_projection.shape[-1]
        
        self._freeze(cfg.FINETUNE_TYPE)
        
    @classmethod
    def from_config(cls, cfg, all_cfg):
        return {
            "cfg": cfg,
        }
        
    
    def _freeze(self, finetune_type: str):
        if finetune_type == 'all':
            return
        elif finetune_type == 'none':
            for p in self.parameters():
                p.requires_grad = False
            return
        elif finetune_type == 'before_proj':
            for p in self.proxy.parameters():
                p.requires_grad = True
            for p in self.proxy.ln_post.parameters():
                p.requires_grad = False
            if self.proxy.proj is not None:
                self.proxy.proj.requires_grad = False
            last_block: ResidualAttentionBlock = self.proxy.transformer.resblocks[-1]
            for p in last_block.ls_1.parameters():
                p.requires_grad = False
            for p in last_block.ls_2.parameters():
                p.requires_grad = False
            for p in last_block.mlp.parameters():
                p.requires_grad = False
            for p in last_block.ln_2.parameters():
                p.requires_grad = False
        elif finetune_type == 'attention':
            for p in self.proxy.parameters():
                p.requires_grad = False
            attn_strs = []
            for name, params in self.proxy.named_parameters():
                if 'attn' in name:
                    if 'in_proj' in name or 'q_proj' in name or 'v_proj' in name:
                        attn_strs.append(name)                        
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif 'position' in name:
                    attn_strs.append(name)
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            _logger.info(f"Finetune only attention in {self.__class__.__name__}: {attn_strs}")
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
        
    def _interpolate_postion_embedding(self, curr_grid_size: Tuple[int, int]) -> Tensor:
        pretrained_grid_size = self.proxy.grid_size
        pretrained_pe = self.proxy.positional_embedding
        if curr_grid_size == pretrained_grid_size:
            return pretrained_pe
        global_pe, local_pe = pretrained_pe[0:1], pretrained_pe[1:]
        local_pe = local_pe.reshape(*pretrained_grid_size, local_pe.shape[-1]).permute(2, 0, 1)
        interp_local_pe = F.interpolate(local_pe.unsqueeze(0), size=curr_grid_size, mode='bicubic', align_corners=True).squeeze(0)
        interp_local_pe = interp_local_pe.permute(1, 2, 0).reshape(-1, interp_local_pe.shape[0])
        interp_pe = torch.cat([global_pe, interp_local_pe], dim=0)
        return interp_pe
    
    def _attention_with_biases(self, block: ResidualAttentionBlock, 
                               x: Tensor, biases: List[Tensor], 
                               former_m_embs: Optional[List[Tensor]]=None) -> Tuple[Tensor, List[Tensor]]:
        L, B = x.shape[:2]
        seq_lens = [b.shape[1] for b in biases]
        max_seq_len = max(seq_lens)
        
        if former_m_embs is None:
            # init with cls token
            former_m_embs = []
            for i, seq_len in enumerate(seq_lens):
                Q = seq_len - L
                former_m_embs.append(x[0:1, i].repeat(Q, 1))
        
        padded_x = x.new_zeros(max_seq_len, B, x.shape[-1])
        padded_attn_mask = biases[0].new_zeros(B, self.neck_heads, max_seq_len, max_seq_len)
        for i, seq_len in enumerate(seq_lens):
            m_emb = former_m_embs[i]
            Q = m_emb.shape[0]
            padded_x[:Q, i] = m_emb
            padded_x[Q:seq_len, i] = x[:, i]
            padded_attn_mask[i, :, :seq_len, :seq_len] = biases[i]
            padded_attn_mask[i, :, :, seq_len:] = -float('inf')
        padded_attn_mask = padded_attn_mask.flatten(0, 1)
        padded_x = block(padded_x, attn_mask=padded_attn_mask)
        m_embs = []
        out_x = []
        for i, seq_len in enumerate(seq_lens):
            Q = seq_len - L
            m_embs.append(padded_x[:Q, i])
            out_x.append(padded_x[Q:seq_len, i])
                
        out_x = torch.stack(out_x, dim=1)
        return out_x, m_embs
            
    
    def _masks_to_attn_biases(self, masks: List[Tensor], target_shape: Tuple[int, int]) -> Tuple[List[Tensor], List[Tensor]]:
        attn_biases = []
        areas = []
        for mask in masks:
            Q = mask.shape[0]
            hw = target_shape[0] * target_shape[1]
            if Q == 0:
                attn_biases.append(torch.zeros(self.neck_layers, self.neck_heads, 1+hw, 1+hw, dtype=mask.dtype, device=mask.device))
                areas.append(torch.zeros(0, dtype=mask.dtype, device=mask.device))
                continue
            down_mask = downsample_masks(mask.float(), target_shape, self._downsample_method)  # Q, H, W
            areas.append(down_mask.sum(dim=(1, 2)))
            if self._down_mask_thresh is None:
                th = down_mask.flatten(1).max(dim=1).values / 2
                th = th[:, None]
            else:
                th = self._down_mask_thresh
            down_mask = down_mask.flatten(1) # Q, H * W
            bin_down_mask = down_mask > th
            attn_bias = torch.zeros(self.neck_layers, self.neck_heads, Q+1+hw, Q+1+hw, dtype=down_mask.dtype, device=down_mask.device)
            minimum = -100  # Actually it should be -inf, but there may be empty masks which would cause NaN after softmax
            down_mask = down_mask[None, None, :, :]
            down_mask = down_mask * self.mask_logit_scale.exp()[:, :, None, None]
            down_mask[:, :, ~bin_down_mask] = minimum
            attn_bias[:, :, Q:, :Q] = -float('inf')
            attn_bias[:, :, :Q, :(Q+1)] = -float('inf')
            attn_bias[:, :, :Q, -hw:].copy_(down_mask)
            attn_biases.append(attn_bias)
        return attn_biases, areas
    
    def _extract_features(self, x: Tensor, masks: List[Tensor]):
        out = {}
        x = self.proxy.conv1(x)  # shape = [*, width, grid, grid]
        curr_grid_size = x.shape[-2:]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.proxy.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self._interpolate_postion_embedding(curr_grid_size)
        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.proxy.patch_dropout(x)
        x = self.proxy.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        attn_biases, areas = self._masks_to_attn_biases(masks, curr_grid_size)
        out[f"areas{self._feature_suffix}"] = areas
        m_embs = None
        for i, block in enumerate(self.proxy.transformer.resblocks, 1):
            if i < self._mask_prior_beg:
                x = block(x)
            else:
                curr_neck_layer = len(self.proxy.transformer.resblocks) - i
                curr_attn_biases = [attn_bias[curr_neck_layer] for attn_bias in attn_biases]
                x, m_embs = self._attention_with_biases(block, x, curr_attn_biases, m_embs)
            fk = f"block{i}{self._feature_suffix}"
            # mi = f"-{len(self.proxy.transformer.resblocks) - i}" if i != len(self.proxy.transformer.resblocks) else ''
            if i == len(self.proxy.transformer.resblocks):
                mi = ''
                proj_m_embs = []
                for m_emb in m_embs:
                    proj_m_emb = self.proxy.ln_post(m_emb)
                    if self.proxy.proj is not None:
                        proj_m_emb = proj_m_emb @ self.proxy.proj
                    proj_m_embs.append(proj_m_emb)
                m_embs = proj_m_embs
            else:
                mi = f"-{len(self.proxy.transformer.resblocks) - i}"
            mk = f"m_embs{mi}{self._feature_suffix}"
            if fk in self._out_features:
                out[fk] = _lnd2ndhw(x, curr_grid_size)
            if mk in self._out_features:
                out[mk] = m_embs
        return out
        
    def extract_features(self, inputs: PaddedList) -> Dict[str, Tensor]:
        if self._finetune_none:
            self.eval()
            with torch.no_grad():
                return self._extract_features(inputs.images, inputs.masks)
        else:
            return self._extract_features(inputs.images, inputs.masks)