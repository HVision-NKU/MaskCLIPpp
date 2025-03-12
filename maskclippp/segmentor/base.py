from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABCMeta, abstractmethod
import logging
import torch
from torch import nn, Tensor, device
import torch.nn.functional as F
from detectron2.utils import comm
from ..vencoder import PaddedList
from ..utils.ckpt import download_mask_generator

_logger = logging.getLogger(__name__)


class BaseSegmentor(nn.Module, metaclass=ABCMeta):
    
    def __init__(self,
                 mask_is_padded,
                 pretrained="") -> None:
        super().__init__()
        self._mask_is_padded = mask_is_padded
        
        if len(pretrained) > 0:
            if comm.get_local_rank() == 0:
                download_mask_generator(pretrained, _logger)
            comm.synchronize()
    
    @abstractmethod
    def is_closed_classifier(self) -> bool:
        raise NotImplementedError("is_closed_classifier is not implemented")
    
    
    @abstractmethod
    def generate_masks(self, 
                       batched_inputs: Dict[str, Any], 
                       encode_dict: Dict[str, Tensor]) -> Tuple[Union[List[Tensor], Tensor], Optional[Union[List[Tensor], Tensor]]]:
        """_summary_

        Args:
            batched_inputs (Dict[str, Any]): If Segmentor use batched_inputs as input, it means the mask is unpadded.
            encode_dict (Dict[str, Tensor]): If Segmentor use features in encode_dict as input, it means the mask is padded.

        Returns:
            Tuple[Union[List[Tensor], Tensor], Optional[Union[List[Tensor], Tensor]]]: Soft masks (List B of Q,H,W) and Class Logits (List B of Q,K)
        """
        pass

    def forward(self, batched_inputs, encode_dict):
        # assert not self.training, "Segmentor should be used in inference mode"
        pred_masks, pred_logits = self.generate_masks(batched_inputs, encode_dict)
        input_f: PaddedList = encode_dict["input_f"]
        if self._mask_is_padded:
            input_f.set_padded_masks(pred_masks)
        else:
            input_f.set_unpadded_masks(pred_masks)
        return pred_logits
    