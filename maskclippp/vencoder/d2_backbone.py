import logging
from typing import Dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from detectron2.utils import comm
import open_clip

import pickle
from detectron2.utils.file_io import PathManager
from detectron2.modeling import ShapeSpec

from detectron2.config import configurable
from detectron2.modeling import build_backbone

from .build import VISUAL_ENCODER_REGISTRY
from .base import BaseVisualEncoder, PaddedList
from ..utils.ckpt import convert_ndarray_to_tensor

logger = logging.getLogger(__name__)

@VISUAL_ENCODER_REGISTRY.register()
class D2Backbone(BaseVisualEncoder):
    @configurable
    def __init__(self, 
                 cfg,
                 proxy: nn.Module):
        super().__init__(cfg)
        pretrained = cfg.PRETRAINED
        self.proxy = proxy
        self._feature_suffix = cfg.FEATURE_SUFFIX
        
        assert pretrained.endswith(".pkl"), f"Only .pkl file is supported. Got {pretrained}"
        with PathManager.open(pretrained, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        ckpt = data['model']
        backbone_dict = {}
        backbone_str = "backbone."
        for k, v in ckpt.items():
            if k.startswith(backbone_str):
                backbone_dict[k[len(backbone_str):]] = v
        convert_ndarray_to_tensor(backbone_dict)
        missing_keys, unexpected_keys = self.proxy.load_state_dict(backbone_dict, strict=False)
        if len(missing_keys) > 0:
            logger.warning("Missing keys when loading pretrained D2Backbone from %s:\n%s",
                           pretrained, missing_keys)
        if len(unexpected_keys) > 0:
            logger.warning("Unexpected keys when loading pretrained D2Backbone from %s:\n%s",
                           pretrained, unexpected_keys)
            
        # just freeze all
        for param in self.proxy.parameters():
            param.requires_grad = False
        logger.info("Loaded pretrained D2Backbone from %s", pretrained)
        
    @classmethod
    def from_config(cls, cfg, all_cfg):
        proxy = build_backbone(all_cfg)
        return {
            "cfg": cfg,
            "proxy": proxy}
    
    def output_shape(self):
        shape_dict = self.proxy.output_shape()
        new_shape_dict = {}
        for k, v in shape_dict.items():
            new_shape_dict[k+self._feature_suffix] = v
        return new_shape_dict
    
    def _extract_features(self, x):
        features = self.proxy(x)
        out_features = {}
        for k, v in features.items():
            out_features[k+self._feature_suffix] = v
        return out_features
    
    def extract_features(self, inputs: PaddedList) -> Dict[str, Tensor]:
        if self._finetune_none:
            self.eval()
            with torch.no_grad():
                return self._extract_features(inputs.images)
        else:
            return self._extract_features(inputs.images)