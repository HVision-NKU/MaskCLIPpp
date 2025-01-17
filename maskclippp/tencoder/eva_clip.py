from typing import List, Optional
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from detectron2.utils import comm
import eva_clip
from eva_clip.transformer import TextTransformer

_logger = logging.getLogger(__name__)

from .build import TEXT_ENCODER_REGISTRY
from ..utils.ckpt import load_state_dict_with_beg_key


def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
        if pool_type == 'first':
            pooled, tokens = x[:, 0], x[:, 1:]
        elif pool_type == 'last':
            pooled, tokens = x[:, -1], x[:, :-1]
        elif pool_type == 'argmax':
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            assert text is not None
            pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
        else:
            pooled = tokens = x
        return pooled, tokens


@TEXT_ENCODER_REGISTRY.register()
class EVACLIPTextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_name = cfg.MODEL_NAME
        pretrained = cfg.PRETRAINED
        
        clip_model = eva_clip.create_model_and_transforms(model_name, pretrained=pretrained, force_custom_clip=True)[0]
        self.text_tokenizer = eva_clip.get_tokenizer(model_name)
        assert isinstance(clip_model.text, TextTransformer), type(clip_model.text)
        self.proxy: TextTransformer = clip_model.text
        self.logit_scale = clip_model.logit_scale
        
        if len(cfg.LOAD_FROM) > 0:
            load_from_path = Path(cfg.LOAD_FROM)
            if not load_from_path.exists():
                raise FileNotFoundError(f"Could not find {load_from_path}")
            load_ckpt = torch.load(load_from_path, map_location="cpu")
            if 'model' in load_ckpt:
                load_ckpt = load_ckpt['model']
            load_beg_key = cfg.LOAD_BEG_KEY
            load_state_dict_with_beg_key(self.proxy, load_ckpt, f"{load_beg_key}text.", 
                                         self.__class__.__name__+".proxy", load_from_path, _logger)
            # self.logit_scale. load_ckpt[f"{load_beg_key}logit_scale"]
            if f"{load_beg_key}logit_scale" in load_ckpt:
                self.logit_scale.data = load_ckpt[f"{load_beg_key}logit_scale"]
            else:
                _logger.warning("Missing logit_scale in %s, using default", load_from_path)
        
        if cfg.SKIP_LN_FINAL:
            _logger.warning("SKIP_LN_FINAL has no use in %s", self.__class__.__name__)
        
        self._freeze(cfg.FINETUNE_TYPE)
        self.finetune_none = cfg.FINETUNE_TYPE == 'none'
    
    def _forward(self, text: List[str]):
        tokens = self.text_tokenizer(text).to(self.logit_scale.device)
        return self.proxy(tokens)
        
    def forward(self, text: List[str]):
        if self.finetune_none:
            self.eval()
            with torch.no_grad():
                return self._forward(text)
        else:
            return self._forward(text)
        
    def _freeze(self, finetune_type):
        if finetune_type == 'all':
            for p in self.parameters():
                p.requires_grad = True
        elif finetune_type == 'none':
            for p in self.parameters():
                p.requires_grad = False
        elif finetune_type == 'attention':
            attn_strs = []
            for name, params in self.named_parameters():
                if 'transformer' in name:
                    if 'attn' in name:
                        if 'q_proj_weight' in name or 'v_proj_weight' in name or 'in_proj_weight' in name:
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
        elif finetune_type == 'proj':
            for p in self.parameters():
                p.requires_grad = False
            for p in self.proxy.ln_final.parameters():
                p.requires_grad = True
            self.proxy.text_projection.requires_grad = True
            self.logit_scale.requires_grad = True
        else:
            raise ValueError(f"Unknown finetune type {finetune_type} for {self.__class__.__name__}")