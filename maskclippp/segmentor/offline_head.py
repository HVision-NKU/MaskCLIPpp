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

from .build import SEGMENTOR_REGISTRY
from .base import BaseSegmentor


_logger = logging.getLogger(__name__)


@SEGMENTOR_REGISTRY.register()
class OfflineSegmentor(BaseSegmentor):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        ann_dir: str,
        ann_suffix: str,
    ):
        super().__init__(mask_is_padded=False)
        root_dir = Path(os.environ.get("DETECTRON2_DATASETS", "datasets"))
        self.ann_dir = root_dir / Path(ann_dir)
        self.ann_suffix = ann_suffix
        # self.img_dir = root_dir / "ADEChallengeData2016/images/validation"  # tmp
        assert self.ann_dir.is_dir(), f"Directory {self.ann_dir} does not exist."

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        input_shape =  {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.MASKCLIPPP.SEGMENTOR.IN_FEATURES
        }
        return {
            "input_shape": input_shape,
            "ann_dir": cfg.MODEL.MASKCLIPPP.SEGMENTOR.OFFLINE_ANN_DIR,
            "ann_suffix": cfg.MODEL.MASKCLIPPP.SEGMENTOR.OFFLINE_ANN_SUFFIX,
        }
        
    def _read_masks_from_file(self, ann_path, device) -> Tensor:
        if self.ann_suffix == ".json":
            with open(ann_path, "r") as f:
                anns = json.load(f)
            if len(anns) == 0:
                # _logger.warning("No annotations in %s", ann_path)
                raise NotImplementedError()
                # print(f"No annotations in {ann_path}")
                # img_path = self.img_dir / (Path(ann_path).stem + ".jpg")
                # img_size = Image.open(img_path).size
                # masks = torch.zeros((1, img_size[1], img_size[0]), dtype=torch.uint8, device=device)
            else:
                masks = np.stack([np.asarray(mask_utils.decode(x["segmentation"])) for x in anns], axis=0)
                masks = torch.from_numpy(masks).to(device)
        elif self.ann_suffix == '.png':
            im = np.asarray(Image.open(ann_path))
            if im.ndim == 3:
                im = rgb2id(im)
            label_set = np.unique(im)
            masks = []
            for label in label_set:
                mask = im == label
                masks.append(mask)
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks).to(device)
        elif self.ann_suffix == '.npy':
            semseg = np.load(ann_path)  # (H, W) with label
            label_set = np.unique(semseg)
            masks = []
            for label in label_set:
                mask = semseg == label
                masks.append(mask)
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks).to(device)
        else:
            raise NotImplementedError()
        return masks
    
    def is_closed_classifier(self) -> bool:
        return False
    
    def generate_masks(self, batched_inputs: Dict[str, Any], encode_dict: Dict[str, Tensor]) -> Tuple[List[Tensor] | Tensor | None]:
        fnames = [x["file_name"] for x in batched_inputs]
        base_names = [Path(x).stem for x in fnames]
        ann_paths = [self.ann_dir / f"{x}{self.ann_suffix}" for x in base_names]
        
        device = next(iter(encode_dict.values())).device
        pred_masks = []
        for i, ann_path in enumerate(ann_paths):
            pred_masks.append(self._read_masks_from_file(ann_path, device).float())
        return pred_masks, None
            