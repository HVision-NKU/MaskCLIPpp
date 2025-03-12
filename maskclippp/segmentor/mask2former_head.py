# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import pickle
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.file_io import PathManager

from .build import SEGMENTOR_REGISTRY
from .base import BaseSegmentor
from .mask2former.pixel_decoder import MSDeformAttnPixelDecoder
from .mask2former.transformer_decoder import MultiScaleMaskedTransformerDecoder
from ..utils.ckpt import convert_ndarray_to_tensor

logger = logging.getLogger(__name__)

@SEGMENTOR_REGISTRY.register()
class Mask2FormerSegmentor(BaseSegmentor):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        pretrained: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__(mask_is_padded=True, pretrained=pretrained)
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        
        assert pretrained.endswith(".pkl"), f"Only .pkl file is supported. Got {pretrained}"
        with PathManager.open(pretrained, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        mask2former_ckpt = data['model']
        head_dict = {}
        head_str = "sem_seg_head."
        for k, v in mask2former_ckpt.items():
            if k.startswith(head_str):
                head_dict[k[len(head_str):]] = v
        if "predictor.query_feat.weight" not in head_dict:
            assert "predictor.static_query.weight" in head_dict, "No predictor.query_feat.weight in pretrained model."
            head_dict["predictor.query_feat.weight"] = head_dict.pop("predictor.static_query.weight")
            logger.warning("Use predictor.static_query.weight as predictor.query_feat.weight")
            # logger.warning("Use")
        convert_ndarray_to_tensor(head_dict)
        missing_keys, unexpected_keys = self.load_state_dict(head_dict, strict=False)
        if len(missing_keys) > 0:
            logger.warning("Missing keys when loading pretrained Mask2FormerSegmentor from %s:\n%s",
                           pretrained, missing_keys)
        if len(unexpected_keys) > 0:
            logger.warning("Unexpected keys when loading pretrained Mask2FormerSegmentor from %s:\n%s",
                           pretrained, unexpected_keys)
            
        for param in self.parameters():
            param.requires_grad = False
        
        logger.info("Loaded pretrained Mask2FormerSegmentor from %s", pretrained)
        

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        input_shape =  {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.MASKCLIPPP.SEGMENTOR.IN_FEATURES
        }

        
        pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape,
            transformer_dropout=cfg.MODEL.MASK_FORMER.DROPOUT,
            transformer_nheads=cfg.MODEL.MASK_FORMER.NHEADS,
            # transformer_dim_feedforward=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            transformer_dim_feedforward=1024,  #  use 1024 for deformable transformer encoder
            transformer_enc_layers=cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS,
            conv_dim=cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            mask_dim=cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
            norm=cfg.MODEL.SEM_SEG_HEAD.NORM,
            transformer_in_features=cfg.MODEL.MASKCLIPPP.SEGMENTOR.TRANSFORMER_IN_FEATURES,
            common_stride=cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        )
        
        transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        transformer_predictor = MultiScaleMaskedTransformerDecoder(
            in_channels=transformer_predictor_in_channels,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            hidden_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            num_queries=cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            nheads=cfg.MODEL.MASK_FORMER.NHEADS,
            dim_feedforward=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            dec_layers=cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1,
            pre_norm=cfg.MODEL.MASK_FORMER.PRE_NORM,
            enforce_input_project=cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ,
            mask_dim=cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
        )
        
        return {
            "input_shape": input_shape,
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": pixel_decoder,
            "transformer_predictor": transformer_predictor,
            "pretrained": cfg.MODEL.MASKCLIPPP.SEGMENTOR.PRETRAINED,
        }

    def is_closed_classifier(self) -> bool:
        return True
    
    def generate_masks(self, batched_inputs: Dict[str, Any], encode_dict: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor] | torch.Tensor | None]:
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(encode_dict)
        predictions = self.predictor(multi_scale_features, mask_features, None)
        pred_masks = predictions["pred_masks"].sigmoid()
        pred_logits = predictions["pred_logits"]
        return pred_masks, pred_logits
