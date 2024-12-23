from detectron2.utils.registry import Registry
from .base import BaseVisualEncoder

VISUAL_ENCODER_REGISTRY = Registry("VISUAL_ENCODER")

def build_visual_encoder(cfg, all_cfg) -> BaseVisualEncoder:
    visual_encoder_name = cfg.get("NAME", "none")
    if visual_encoder_name.lower() == "none":
        return None
    visual_encoder = VISUAL_ENCODER_REGISTRY.get(visual_encoder_name)(cfg, all_cfg)
    return visual_encoder