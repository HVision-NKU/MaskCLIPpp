from typing import List, Tuple, Dict, Union, Optional

import logging
from torch import nn, Tensor

from detectron2.config import configurable
from detectron2.layers import ShapeSpec


from .build import PSM_REGISTRY


logger = logging.getLogger(__name__)


@PSM_REGISTRY.register()
class LinearPSM(nn.Module):
    @configurable
    def __init__(self,
                 input_shape: Dict[str, ShapeSpec],
                 template_width: int,
                 corr_width: int):
        super().__init__()
        self.corr_proj_in = nn.Sequential(
            nn.Linear(template_width, corr_width),
        )
        self.corr_proj_out = nn.Sequential(
            nn.Linear(corr_width, 1)
        )
        
    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec], num_templates):
        return {
            "input_shape": input_shape,
            "template_width": num_templates,
            "corr_width": cfg.CORR_WIDTH,
        }
        
    def forward(self, corrs: Tensor, masks: Tensor, mask2batch: Tensor, features: Dict[str, Tensor]):
        N, Ka = corrs.shape[:2]
        corrs = self.corr_proj_in(corrs)
        corrs = self.corr_proj_out(corrs)  # N,Ka,1
        corrs = corrs.squeeze(-1)
        return corrs
        