from typing import Dict
from detectron2.layers import ShapeSpec
from detectron2.config import configurable
from torch import Tensor
from .base import BaseVisualEncoder, PaddedList
from .build import VISUAL_ENCODER_REGISTRY


@VISUAL_ENCODER_REGISTRY.register()
class DummyVisualEncoder(BaseVisualEncoder):
    @configurable
    def __init__(self, cfg):
        super().__init__(cfg)
    
    @classmethod
    def from_config(cls, cfg, all_cfg):
        return {"cfg": cfg}

    def extract_features(self, inputs: PaddedList) -> Dict[str, Tensor]:
        return {}
    
    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {}