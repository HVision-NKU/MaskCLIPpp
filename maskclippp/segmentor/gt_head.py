import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import os
import json
import numpy as np
from pathlib import Path
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from PIL import Image

import pycocotools.mask as mask_utils
from panopticapi.utils import rgb2id

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

from .build import SEGMENTOR_REGISTRY
from .base import BaseSegmentor


_logger = logging.getLogger(__name__)


@SEGMENTOR_REGISTRY.register()
class GTSegmentor(BaseSegmentor):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        dataset_name: str,
    ):
        super().__init__(mask_is_padded=False)
        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }
        meta = MetadataCatalog.get(dataset_name)
        self._ignore_label = meta.ignore_label
        
        
    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        input_shape =  {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.MASKCLIPPP.SEGMENTOR.IN_FEATURES
        }
        dataset_name = cfg.DATASETS.TEST[0]
        return {
            "input_shape": input_shape,
            "dataset_name": dataset_name,
        }
    
    def is_closed_classifier(self) -> bool:
        return False
    
    
    def _load_sem_seg_ann(self, ann_path, dtype):
        with PathManager.open(ann_path, "rb") as f:
            array = np.asarray(Image.open(f), dtype=dtype)
        return array
        
    
    def generate_masks(self, batched_inputs: Dict[str, Any], encode_dict: Dict[str, Tensor]) -> Tuple[List[Tensor] | Tensor | None]:
        fnames = [x["file_name"] for x in batched_inputs]
        ann_paths = [self.input_file_to_gt_file[fname] for fname in fnames]
        device = next(iter(encode_dict.values())).device
        pred_masks = []
        for i, ann_path in enumerate(ann_paths):
            gt_ann = self._load_sem_seg_ann(ann_path, int)
            gt_classes = np.unique(gt_ann)
            gt_classes = gt_classes[gt_classes != self._ignore_label]
            masks = []
            for c in gt_classes:
                mask = gt_ann == c
                masks.append(mask)
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks).to(dtype=torch.float32, device=device)
            pred_masks.append(masks)
        return pred_masks, None
            