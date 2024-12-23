from typing import List, Tuple, Dict, Union, Optional

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from .build import PSM_REGISTRY


@PSM_REGISTRY.register()
class PseudoTextPSM(nn.Module):
    
    @configurable
    def __init__(self, 
                 input_shape: Dict[str, ShapeSpec],
                 template_width: int,
                 corr_width: int,
                 in_feature_keys: List[str],
                 num_heads: int,
                 detach_visual_cond: bool,
                 norm_visual_cond: bool,
                 corr_residual: bool,
                 use_logit_scale: bool,
                 hidden_dropout_prob: float,
                 attention_probs_dropout_prob: float):
        super().__init__()
        self.img_cond_keys = in_feature_keys
        self.detach_visual_cond = detach_visual_cond
        self.norm_visual_cond = norm_visual_cond
        self.corr_residual = corr_residual
        img_width = input_shape[self.img_cond_keys[0]].channels
        if len(self.img_cond_keys) > 1:
            for key in self.img_cond_keys[1:]:
                assert input_shape[key].channels == img_width, f"Input shape {key} should have the same channels as {self.img_cond_keys[0]}"
        text_width = input_shape["emb"].channels
        self.img_proj = nn.Linear(img_width, text_width, bias=False)
        self.img2text = nn.MultiheadAttention(text_width, 
                                              num_heads,
                                              dropout=attention_probs_dropout_prob,
                                              batch_first=True)
        self.ln_1 = nn.LayerNorm(text_width)
        self.dropout_1 = nn.Dropout(hidden_dropout_prob)
        if use_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self.logit_scale = None
        if self.corr_residual:
            self.corrs_in = nn.Linear(template_width, corr_width)
            self.corrs_out = nn.Linear(corr_width, 1)

    
    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec], num_templates):
        return {
            "input_shape": input_shape,
            "template_width": num_templates,
            "corr_width": cfg.CORR_WIDTH,
            "in_feature_keys": cfg.IN_FEATURES,
            "num_heads": cfg.NUM_HEADS,
            "detach_visual_cond": cfg.DETACH_VISUAL_COND,
            "norm_visual_cond": cfg.NORM_VISUAL_COND,
            "corr_residual": cfg.CORR_RESIDUAL,
            "use_logit_scale": cfg.USE_LOGIT_SCALE,
            "hidden_dropout_prob": cfg.HIDDEN_DROPOUT_PROB,
            "attention_probs_dropout_prob": cfg.ATTENTION_PROBS_DROPOUT_PROB,
        }
        
    def forward(self, corrs: Tensor, masks: Tensor, mask2batch: Tensor, features: Dict[str, Tensor]):
        img_cond_list = []
        for key in self.img_cond_keys:
            img_cond = features[key]
            if self.detach_visual_cond:
                img_cond = img_cond.detach()
            img_cond = img_cond.flatten(2).transpose(1, 2)
            img_cond_list.append(img_cond)
        img_cond = torch.cat(img_cond_list, dim=1)
        
        if self.norm_visual_cond:
            img_cond = F.normalize(img_cond, dim=-1)
        img_cond = self.img_proj(img_cond)  # B,L,D
        
        t_embs = features["t_embs"]  # K,T,D, normed
        t_embs = F.normalize(t_embs.mean(dim=1), dim=-1)  # K,D
        t_embs = t_embs.unsqueeze(0).expand(img_cond.size(0), -1, -1)  # B,K,D
        t_embs = self.ln_1(t_embs + self.dropout_1(self.img2text(t_embs, img_cond, img_cond)[0]))
        m_embs = torch.cat(features["m_embs"], dim=0)  # N,D, not normed
        m_embs = F.normalize(m_embs, dim=-1)
        
        bids = torch.unique(mask2batch)
        corrs_rtn = []
        for bid in bids:
            m_embs_per_img = m_embs[mask2batch == bid]
            corrs_per_img = torch.einsum("qd,kd->qk", m_embs_per_img, t_embs[bid])
            corrs_rtn.append(corrs_per_img)
        
        corrs_rtn = torch.cat(corrs_rtn, dim=0)
        if self.logit_scale is not None:
            corrs_rtn = corrs_rtn * torch.clamp(self.logit_scale.exp(), max=100)
        if self.corr_residual:
            corrs = self.corrs_out(self.corrs_in(corrs)).squeeze(-1)
            corrs_rtn += corrs
        return corrs_rtn