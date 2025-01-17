from typing import List, Optional
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from detectron2.utils import comm
import open_clip

from .build import TEXT_ENCODER_REGISTRY
from ..utils.ckpt import load_state_dict_with_beg_key

_logger = logging.getLogger(__name__)


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
class CLIPTextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_name = cfg.MODEL_NAME
        pretrained = cfg.PRETRAINED
        
        clip_model = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)[0]
        self.text_tokenizer = open_clip.get_tokenizer(model_name)
        
        self.transformer = clip_model.transformer
        self.vocab_size = clip_model.vocab_size
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.text_pool_type = getattr(clip_model, 'text_pool_type', 'argmax')
        self.logit_scale = clip_model.logit_scale
        self.register_buffer("attn_mask", clip_model.attn_mask, persistent=False)
        
        if len(cfg.LOAD_FROM) > 0:
            load_from_path = Path(cfg.LOAD_FROM)
            if not load_from_path.exists():
                raise FileNotFoundError(f"Could not find {load_from_path}")
            load_ckpt = torch.load(load_from_path, map_location="cpu")
            if 'model' in load_ckpt:
                load_ckpt = load_ckpt['model']
            load_beg_key = cfg.LOAD_BEG_KEY
            load_state_dict_with_beg_key(self, load_ckpt, load_beg_key, 
                                         self.__class__.__name__, load_from_path, _logger)
        
        self._freeze(cfg.FINETUNE_TYPE)
        self.finetune_none = cfg.FINETUNE_TYPE == 'none'
    
    def _forward(self, text: List[str]):
        tokens = self.text_tokenizer(text).to(self.attn_mask.device)
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(tokens).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x, _ = text_global_pool(x, text=tokens, pool_type=self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection
        return x
    
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
                    else:
                        params.requires_grad = False
                elif 'position' in name:
                    attn_strs.append(name)
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            _logger.info("Finetune only attention in %s: %s",
                         self.__class__.__name__, attn_strs)

        elif finetune_type == 'proj':
            for p in self.parameters():
                p.requires_grad = False
            for p in self.ln_final.parameters():
                p.requires_grad = True
            self.text_projection.requires_grad = True
            self.logit_scale.requires_grad = True
        else:
            raise ValueError(f"Unknown finetune type {finetune_type} for {self.__class__.__name__}")
    