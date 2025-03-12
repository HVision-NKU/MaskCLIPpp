from typing import Tuple, List, Optional, Dict
import logging
import os
from pathlib import Path
from functools import partial
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils import comm
import eva_clip
from eva_clip.eva_vit_model import Block, EVAVisionTransformer, Attention

from detectron2.config import configurable
from detectron2.modeling import ShapeSpec
from .build import VISUAL_ENCODER_REGISTRY
from .base import BaseVisualEncoder, PaddedList
from ..utils.misc import downsample_masks
from ..utils.ckpt import load_state_dict_with_beg_key

_logger = logging.getLogger(__name__)


def _nld2ndhw(x, grid_size):
    return x[:, 1:].transpose(1, 2).reshape(x.shape[0], x.shape[-1], *grid_size).contiguous()


@VISUAL_ENCODER_REGISTRY.register()
class EVACLIPViT(BaseVisualEncoder):
    @configurable
    def __init__(self, cfg):
        BaseVisualEncoder.__init__(self, cfg)
        model_name = cfg.MODEL_NAME
        pretrained = cfg.PRETRAINED
        
        # download on local rank 0 first
        if comm.get_local_rank() == 0:
            eva_clip.create_model_and_transforms(model_name, pretrained=pretrained, force_custom_clip=True)
        comm.synchronize()
        
        clip_model = eva_clip.create_model_and_transforms(model_name, pretrained=pretrained, force_custom_clip=True)[0]
        
        self.proxy: EVAVisionTransformer = clip_model.visual
            
        if len(cfg.LOAD_FROM) > 0:
            load_from_path = Path(cfg.LOAD_FROM)
            if not load_from_path.exists():
                raise FileNotFoundError(f"LOAD_FROM {load_from_path} does not exist")
            load_ckpt = torch.load(load_from_path, map_location='cpu')
            if 'model' in load_ckpt:
                load_ckpt = load_ckpt['model']
            load_state_dict_with_beg_key(self.proxy, load_ckpt, cfg.LOAD_BEG_KEY, 
                                        self.__class__.__name__+".proxy", load_from_path, _logger)
        
        assert self.proxy.rel_pos_bias is None
        assert self.proxy.fc_norm is None
        
        self._out_features = cfg.OUT_FEATURES
        
        self._mask_prior_beg = cfg.MASK_PRIOR_BEG
        self._downsample_method = cfg.DOWNSAMPLE_METHOD
        self._down_mask_thresh = cfg.DOWN_MASK_THRESH
        
        self.neck_layers = len(self.proxy.blocks) - self._mask_prior_beg + 1
        self.neck_heads = self.proxy.blocks[-1].attn.num_heads
        init_mask_logit_scale = torch.full((self.neck_layers, self.neck_heads), cfg.MASK_LOGIT_SCALE)
        if cfg.LEARNABLE_MASK_LOGIT_SCALE:
            self.register_parameter("mask_logit_scale", nn.Parameter(init_mask_logit_scale))
        else:
            self.register_buffer("mask_logit_scale", init_mask_logit_scale, persistent=False)
        
        assert 0 < self._mask_prior_beg <= len(self.proxy.blocks), f"Invalid mask prior begin index {self._mask_prior_beg}, should be in [1, {len(self.proxy.transformer.resblocks)}]"
        self._output_strides = {}
        self._output_channels = {}
        
        origin_grid_size = self.proxy.patch_embed.patch_size[0]
        embed_dim = self.proxy.embed_dim
        for i in range(1, len(self.proxy.blocks)):
            self._output_strides[f"block{i}{self._feature_suffix}"] = origin_grid_size
            self._output_channels[f"block{i}{self._feature_suffix}"] = embed_dim
            mi = f'-{len(self.proxy.blocks) - i}'
            self._output_strides[f"m_embs{mi}{self._feature_suffix}"] = -1
            self._output_channels[f"m_embs{mi}{self._feature_suffix}"] = embed_dim
            self._output_strides[f"i_embs{i}{self._feature_suffix}"] = -1
            self._output_channels[f"i_embs{i}{self._feature_suffix}"] = embed_dim
        self._output_strides[f"emb{self._feature_suffix}"] = -1
        self._output_channels[f"emb{self._feature_suffix}"] = self.proxy.head.weight.shape[0]
        self._output_strides[f"m_embs{self._feature_suffix}"] = -1
        self._output_channels[f"m_embs{self._feature_suffix}"] = self.proxy.head.weight.shape[0]
        self._output_strides[f"p_embs{self._feature_suffix}"] = origin_grid_size
        self._output_channels[f"p_embs{self._feature_suffix}"] = self.proxy.head.weight.shape[0]
        self._output_strides[f"i_embs{self._feature_suffix}"] = -1
        self._output_channels[f"i_embs{self._feature_suffix}"] = self.proxy.head.weight.shape[0]
        
        self._pretrained_grid_size = (self.proxy.patch_embed.img_size[0] // self.proxy.patch_embed.patch_size[0], 
                                      self.proxy.patch_embed.img_size[1] // self.proxy.patch_embed.patch_size[1])
        self._freeze(cfg.FINETUNE_TYPE)
        
    @classmethod
    def from_config(cls, cfg, all_cfg):
        return {
            "cfg": cfg,
        }
        
    
    def _freeze(self, finetune_type: str):
        if finetune_type == 'all':
            for p in self.proxy.parameters():
                p.requires_grad = True
        elif finetune_type == 'none':
            for p in self.parameters():
                p.requires_grad = False
            return
        elif finetune_type == 'before_proj':
            for p in self.proxy.parameters():
                p.requires_grad = True
            for p in self.proxy.head.parameters():
                p.requires_grad = False
            for p in self.proxy.norm.parameters():
                p.requires_grad = False
            last_block: Block = self.proxy.blocks[-1]
            for p in last_block.norm2.parameters():
                p.requires_grad = False
            for p in last_block.mlp.parameters():
                p.requires_grad = False
            for p in last_block.attn.inner_attn_ln.parameters():
                p.requires_grad = False
            for p in last_block.attn.proj.parameters():
                p.requires_grad = False
            if last_block.postnorm:
                for p in last_block.norm1.parameters():
                    p.requires_grad = False
            if last_block.gamma_1 is not None:
                last_block.gamma_1.requires_grad = False
                last_block.gamma_2.requires_grad = False                    
        elif finetune_type == 'attention':
            attn_str = []
            for name, params in self.proxy.named_parameters():
                if 'attn' in name:
                    if 'q_proj' in name or 'v_proj' in name or 'qkv' in name:
                        attn_str.append(name)
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif 'pos' in name  or 'position' in name:
                    attn_str.append(name)
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            _logger.info(f"Finetune only attention in {self.__class__.__name__}: {attn_str}")
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
        pretrained_grid_size = self._pretrained_grid_size
        pretrained_pe = self.proxy.pos_embed
        if curr_grid_size == pretrained_grid_size:
            return pretrained_pe
        global_pe, local_pe = pretrained_pe[:, 0:1], pretrained_pe[:, 1:]
        local_pe = local_pe.reshape(local_pe.shape[0], *pretrained_grid_size, local_pe.shape[-1]).permute(0, 3, 1, 2)
        interp_local_pe = F.interpolate(local_pe, size=curr_grid_size, mode='bicubic', align_corners=True)
        interp_local_pe = interp_local_pe.permute(0, 2, 3, 1).reshape(interp_local_pe.shape[0], -1, interp_local_pe.shape[1])
        interp_pe = torch.cat([global_pe, interp_local_pe], dim=1)
        return interp_pe
    
    def _attention_with_biases(self, block: Block, 
                               x: Tensor, biases: List[Tensor], 
                               former_m_embs: Optional[List[Tensor]]=None,
                               return_p_embs: bool=False) -> Tuple[Tensor, List[Tensor], Optional[Tensor]]:
        B, L = x.shape[:2]   # L=1+hw
        # cal p_embs
        if return_p_embs:
            p_embs = self._block_without_attn(block, x)        
        else:
            p_embs = None
        # cal m_embs
        seq_lens = [b.shape[1] for b in biases]
        max_seq_len = max(seq_lens)
        
        if former_m_embs is None:
            # init with cls token
            former_m_embs = []
            for i, seq_len in enumerate(seq_lens):
                Q = seq_len - L
                former_m_embs.append(x[0:1, i].repeat(Q, 1))
            
        padded_attn_mask = biases[0].new_zeros(B, self.neck_heads, max_seq_len, max_seq_len)
        padded_x = x.new_zeros(B, max_seq_len, x.shape[-1])
        
        for i, seq_len in enumerate(seq_lens):
            m_emb = former_m_embs[i]
            Q = m_emb.shape[0]
            padded_x[i, :Q] = m_emb
            padded_x[i, Q:seq_len] = x[i]
            padded_attn_mask[i, :, :seq_len, :seq_len] = biases[i]
            padded_attn_mask[i, :, :, seq_len:] = -float('inf')
        padded_x = block(padded_x, attn_mask=padded_attn_mask)
        m_embs = []
        out_x = []
        for i, seq_len in enumerate(seq_lens):
            Q = seq_len - L
            m_embs.append(padded_x[i, :Q])
            out_x.append(padded_x[i, Q:seq_len])
        out_x = torch.stack(out_x, dim=0)
        return out_x, m_embs, p_embs
            
    def _attention_as_ffn(self, ori_attn: Attention, x: Tensor):
        B, N, C = x.shape
        if ori_attn.subln:
            v = F.linear(input=x, weight=ori_attn.v_proj.weight, bias=ori_attn.v_bias)
            v = v.reshape(B, N, ori_attn.num_heads, -1).permute(0, 2, 1, 3) 
        else:
            qkv_bias = None
            if ori_attn.q_bias is not None:
                qkv_bias = torch.cat((ori_attn.q_bias, torch.zeros_like(ori_attn.v_bias, requires_grad=False), ori_attn.v_bias))
            qkv = F.linear(input=x, weight=ori_attn.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, ori_attn.num_heads, -1).permute(2, 0, 3, 1, 4)   # 3, B, num_heads, N, C
            v = qkv[2]
        v = v.transpose(1, 2).reshape(B, N, -1)
        v = ori_attn.inner_attn_ln(v)
        v = ori_attn.proj(v)
        v = ori_attn.proj_drop(v)
        return v
    
    def _block_without_attn(self, block: Block, x: Tensor):
        if block.gamma_1 is None:
            if block.postnorm:
                v = self._attention_as_ffn(block.attn, x)
                v = x + block.drop_path(block.norm1(v))
                v = v + block.drop_path(block.norm2(block.mlp(v)))
            else:
                v = self._attention_as_ffn(block.attn, block.norm1(x))
                v = x + block.drop_path(v)
                v = v + block.drop_path(block.mlp(block.norm2(v)))
        else:
            if block.postnorm:
                v = self._attention_as_ffn(block.attn, x)
                v = x + block.drop_path(block.gamma_1 * block.norm1(v))
                v = v + block.drop_path(block.gamma_2 * block.norm2(block.mlp(v)))
            else:
                v = self._attention_as_ffn(block.attn, block.norm1(x))
                v = x + block.drop_path(block.gamma_1 * v)
                v = v + block.drop_path(block.gamma_2 * block.mlp(block.norm2(v)))
        return v
            
            
    def _masks_to_attn_biases(self, masks: List[Tensor], target_shape: Tuple[int, int]) -> Tuple[List[Tensor], List[Tensor]]:
        """Convert masks to attention biases.

        Args:
            masks (List[Tensor]): List(B) of masks with shape (Q, H, W)
            target_shape (Tuple[int, int]): target shape of the feat (h, w)

        Returns:
            List[Tensor]: List(B) of attention biases with shape (L, nh, Q+1+hw, Q+1+hw)
            List[Tensor]: List(B) of areas with shape (Q)
        """
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
        H, W = x.shape[-2:]
        curr_grid_size = (H // self.proxy.patch_embed.patch_size[0],
                          W // self.proxy.patch_embed.patch_size[1])
        
        x = self.proxy.patch_embed.proj(x).flatten(2).transpose(1, 2)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.proxy.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.proxy.pos_embed is not None:
            x = x + self._interpolate_postion_embedding(curr_grid_size)
        x = self.proxy.pos_drop(x)
        if os.getenv('RoPE') == '1':
            self.proxy.rope.reset_ft_feq_len(*curr_grid_size)
            if self.training and not isinstance(self.proxy.patch_dropout, nn.Identity):
                x, patch_indices_keep = self.proxy.patch_dropout(x)
                self.proxy.rope.forward = partial(self.proxy.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                self.proxy.rope.forward = partial(self.proxy.rope.forward, patch_indices_keep=None)
                x = self.proxy.patch_dropout(x)
        else:
            x = self.proxy.patch_dropout(x)
        
        attn_biases, areas = self._masks_to_attn_biases(masks, curr_grid_size)
        out[f"areas{self._feature_suffix}"] = areas
        m_embs = None
        pk = f"p_embs{self._feature_suffix}"
        for i, block in enumerate(self.proxy.blocks, 1):
            mi = '' if i == len(self.proxy.blocks) else f"-{len(self.proxy.blocks) - i}"
            mk = f"m_embs{mi}{self._feature_suffix}"
            fk = f"block{i}{self._feature_suffix}"
            ik = f"i_embs{i}{self._feature_suffix}"
            ik_final = f"i_embs{self._feature_suffix}"
            return_p_embs = (i == len(self.proxy.blocks)) and (pk in self._out_features)
            if i < self._mask_prior_beg and (mk not in self._out_features):
                x = block(x)
            else:
                curr_neck_layer = len(self.proxy.blocks) - i
                curr_attn_biases = [attn_bias[curr_neck_layer] for attn_bias in attn_biases]
                x, m_embs, p_embs = self._attention_with_biases(block, x, curr_attn_biases, m_embs, return_p_embs=return_p_embs)
            if i == len(self.proxy.blocks):
                proj_m_embs = []
                for m_emb in m_embs:
                    proj_m_embs.append(self.proxy.head(self.proxy.norm(m_emb)))
                m_embs = proj_m_embs
            if fk in self._out_features:
                out[fk] = _nld2ndhw(x, curr_grid_size)
            if ik in self._out_features:
                out[ik] = x[:, 0].contiguous()
            if i == len(self.proxy.blocks) and ik_final in self._out_features:
                i_embs = self.proxy.head(self.proxy.norm(x[:, 0]))
                out[ik_final] = i_embs[:, :, None, None]
            if mk in self._out_features:
                out[mk] = m_embs
            if i < self._mask_prior_beg:
                m_embs = None
            if return_p_embs:
                out[pk] = _nld2ndhw(self.proxy.head(self.proxy.norm(p_embs)), curr_grid_size)
        return out
        
    def extract_features(self, inputs: PaddedList) -> Dict[str, Tensor]:
        if self._finetune_none:
            self.eval()
            with torch.no_grad():
                return self._extract_features(inputs.images, inputs.masks)
        else:
            return self._extract_features(inputs.images, inputs.masks)