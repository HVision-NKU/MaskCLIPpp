from detectron2.utils.registry import Registry
from .base import BaseSegmentor

SEGMENTOR_REGISTRY = Registry("SEGMENTOR")

def build_segmentor(cfg, input_shape) -> BaseSegmentor:
    segmentor_name = cfg.MODEL.MASKCLIPPP.SEGMENTOR.NAME
    segmentor = SEGMENTOR_REGISTRY.get(segmentor_name)(cfg, input_shape)
    return segmentor