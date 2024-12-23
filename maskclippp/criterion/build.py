from detectron2.utils.registry import Registry
from .base import BaseCriterion

CRITERION_REGISTRY = Registry("CRITERION")

def build_criterion(cfg) -> BaseCriterion:
    criterion_name = cfg.NAME
    criterion = CRITERION_REGISTRY.get(criterion_name)(cfg)
    return criterion