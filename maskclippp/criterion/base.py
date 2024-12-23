from typing import Optional, Dict, Any
from abc import ABCMeta, abstractmethod
import torch
from torch import nn, Tensor


class BaseCriterion(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def forward(self, 
                pred_dict: Dict[str, Any],
                target_dict: Dict[str, Any]) -> Dict[str, Tensor]:
        """_summary_

        Args:
            Keys in pred_dict:
                corrs (Tensor): N,K
                mask2batch (Tensor): N, 0-based batch index
                soft_corrs (Optional[Tensor], optional): N,K. Defaults to None.
                pred_qualities (Optional[Tensor], optional): N. Defaults to None.
                Other keys: in encode_dict
            Keys in target_dict:
                gt_labels (Tensor): N
                gt_soft_labels (Optional[Tensor], optional): N,K. Defaults to None.
                gt_valids (Optional[Tensor], optional): N,K. Binary Tensor. Defaults to None.
                soft_valids (Optional[Tensor], optional): N,K. Binary Tensor. Defaults to None.
        
        Returns:
            Dict[str, Tensor]: loss dict
        """
        pass