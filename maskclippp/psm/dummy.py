from typing import Dict
from torch import nn, Tensor


from .build import PSM_REGISTRY


@PSM_REGISTRY.register()
class DummyPSM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, corrs: Tensor, masks: Tensor, mask2batch: Tensor, features: Dict[str, Tensor]):
        return corrs.mean(dim=-1)