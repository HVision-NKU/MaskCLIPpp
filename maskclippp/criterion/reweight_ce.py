from typing import Optional, Dict, Any
from itertools import chain
import logging
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.utils import comm
from detectron2.config import configurable

from .build import CRITERION_REGISTRY
from .base import BaseCriterion


_logger = logging.getLogger(__name__)


@CRITERION_REGISTRY.register()
class ReweightCELoss(BaseCriterion):
    @configurable
    def __init__(self,
                 temperature: float,
                 balance_cls: bool,
                 ignore_nan: bool,
                 ignore_index: int,
                 ignore_empty: bool):
        super().__init__()
        self.temperature = temperature
        self.balance_cls = balance_cls
        self.ignore_nan = ignore_nan
        self.ignore_index = ignore_index
        self.ignore_empty = ignore_empty
        
    @classmethod
    def from_config(cls, cfg):
        return {
            "temperature": cfg.TEMPERATURE,
            "balance_cls": cfg.BALANCE_CLS,
            "ignore_nan": cfg.IGNORE_NAN,
            "ignore_index": cfg.IGNORE_INDEX,
            "ignore_empty": cfg.IGNORE_EMPTY,
        }
            
    def _cal_ce_loss(self, logits: Tensor, labels: Tensor, mask2batch: Tensor) -> Tensor:
        if self.balance_cls:
            bids = torch.unique(mask2batch)
            local_batch_size = len(bids)
            losses_per_sample = F.cross_entropy(logits/self.temperature, labels, ignore_index=self.ignore_index, reduction="none")
            weights_per_sample = torch.zeros_like(losses_per_sample)
            for bid in bids:
                indices = (mask2batch == bid).nonzero().squeeze(1)
                labels_per_img = labels[indices]
                unique_labels, counts = labels_per_img.unique(return_counts=True)
                num_classes_in_img = len(unique_labels)
                for label, count in zip(unique_labels, counts):
                    same_class_indices = indices[labels_per_img == label]
                    weights_per_sample[same_class_indices] = 1 / (local_batch_size * num_classes_in_img * count)
            loss = (losses_per_sample * weights_per_sample).sum()
        else:
            bids, counts = torch.unique(mask2batch, return_counts=True)
            local_batch_size = len(bids)
            losses_per_sample = F.cross_entropy(logits/self.temperature, labels, ignore_index=self.ignore_index, reduction="none")
            weights_per_sample = torch.zeros_like(losses_per_sample)
            for bid, count in zip(bids, counts):
                weights_per_sample[mask2batch == bid] = 1 / (local_batch_size * count)
            loss = (losses_per_sample * weights_per_sample).sum()
        return loss
        
        
    def forward(self,
                pred_dict: Dict[str, Any],
                target_dict: Dict[str, Any]) -> Dict[str, Tensor]:
        corrs = pred_dict["corrs"]
        mask2batch = pred_dict["mask2batch"]
        gt_labels = target_dict["gt_labels"]
        if corrs.size(0) == 0:
            return corrs.new_zeros([])
        
        if self.ignore_empty:
            areas = pred_dict["areas"]
            areas = torch.cat(areas, dim=0)
            is_empties = areas <= 0
            corrs = corrs[~is_empties]
            gt_labels = gt_labels[~is_empties]
            mask2batch = mask2batch[~is_empties]
        
        loss = self._cal_ce_loss(corrs, gt_labels, mask2batch)
        if self.ignore_nan:
            if torch.isnan(loss).any():
                _logger.warning("NaN detected in CELoss")
                loss = torch.nan_to_num(loss, nan=0.0)
        return loss
