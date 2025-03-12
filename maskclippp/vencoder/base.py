from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from abc import ABCMeta, abstractmethod
import torch
from torch import nn, Tensor, device
import torch.nn.functional as F
from detectron2.layers import ShapeSpec
from detectron2.utils import comm
from ..utils.ckpt import download_mask_generator

_logger = logging.getLogger(__name__)

class PaddedList:
    def __init__(self, images: Tensor, masks: Optional[List[Tensor]], image_sizes: List[Tuple[int, int]]):
        """_summary_

        Args:
            images (Tensor): of shape B,C,H,W
            masks (Optional[List[Tensor]]): List(B) of soft masks(value in [0,1]) of shape Q,H,W, Q may be 0.
            image_sizes (List[Tuple[int, int]]): Each tuple is (h, w). It can be smaller than (H, W) due to padding.
        """
        self.images = images
        self.masks = masks
        self.image_sizes = [torch.Size(x) for x in image_sizes]
        
    def __len__(self) -> int:
        return len(self.image_sizes)

    @property
    def device(self) -> device:
        return self.images.device

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "PaddedList":
        cast_images = self.images.to(*args, **kwargs)
        cast_masks = [mask.to(*args, **kwargs) for mask in self.masks] if self.masks is not None else None
        return PaddedList(cast_images, cast_masks, self.image_sizes)
    
    def __getitem__(self, idx) -> Dict[str, Tensor]:
        size = self.image_sizes[idx]
        rtn_dict = {}
        if self.images is not None:
            rtn_dict['images'] = self.images[idx, :, :size[0], :size[1]]
        if self.masks is not None:
            rtn_dict['masks'] = self.masks[idx][:, :size[0], :size[1]]
        return rtn_dict
    
    def get_unpadded_masks(self) -> List[Tensor]:
        if self.masks is not None:
            return [mask[:, :size[0], :size[1]] for mask, size in zip(self.masks, self.image_sizes)]
        return None

    @staticmethod
    def from_tensors(
        images: List[Tensor],
        masks: Optional[List[Tensor]]=None,
        pad_value: Union[float, Tensor]=0.0,
        mask_pad_value: int = 0,
        size_divisibility: int = 0,
    ) -> "PaddedList":
        assert len(images) > 0
        if masks is not None:
            assert len(images) == len(masks)
            # assert mask and img has same shape
            for i, mask in enumerate(masks):
                assert mask.shape[-2:] == images[i].shape[-2:], f"mask shape {mask.shape} != image shape {images[i].shape}"
      
        image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        image_sizes_tensor = [torch.as_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values
        
        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size = (max_size + (stride - 1)).div(stride, rounding_mode="floor") * stride
        
        batched_imgs = torch.ones(len(images), images[0].shape[0], *max_size, 
                                  device=images[0].device, dtype=images[0].dtype) * pad_value
        for i, img in enumerate(images):
            batched_imgs[i, :, :img.shape[-2], :img.shape[-1]].copy_(img)
            
        if masks is not None:
            batched_masks = []
            for i, mask in enumerate(masks):
                if mask.shape[0] == 0:
                    batched_masks.append(mask.new_zeros((0, *max_size)))
                    continue
                image_size = image_sizes[i]
                pad_size = [0, max_size[1] - image_size[1], 0, max_size[0] - image_size[0]]
                batched_masks.append(F.pad(mask, pad_size, value=mask_pad_value))
        else:
            batched_masks = None
        return PaddedList(batched_imgs, batched_masks, image_sizes)
        
    def set_unpadded_masks(self, masks: List[Tensor], mask_pad_value: int = 0):
        assert len(masks) == len(self)
        padded_size = self.images.shape[-2:]
        padded_masks = []
        for i, mask in enumerate(masks):
            if mask.shape[0] == 0:
                padded_masks.append(mask.new_zeros((0, *padded_size)))
            else:
                image_size = self.image_sizes[i]
                if mask.shape[-2:] == image_size:
                    resized_mask = mask
                else:
                    resized_mask = F.interpolate(mask[None], size=image_size, mode='bilinear')[0]
                pad_size = [0, padded_size[1] - resized_mask.shape[-1], 0, padded_size[0] - resized_mask.shape[-2]]
                padded_masks.append(F.pad(resized_mask, pad_size, value=mask_pad_value))
        self.masks = padded_masks
        return self.masks
        
    def set_padded_masks(self, masks: List[Tensor]):
        assert len(masks) == len(self)
        padded_size = self.images.shape[-2:]
        padded_masks = []
        for mask in masks:
            if mask.shape[0] == 0:
                padded_masks.append(mask.new_zeros((0, *padded_size)))
            else:
                if mask.shape[-2:] == padded_size:
                    padded_masks.append(mask)
                else:
                    padded_masks.append(F.interpolate(mask[None], size=padded_size, mode='bilinear')[0])
        self.masks = padded_masks
        return self.masks

    def get_padded_masks(self) -> List[Tensor]:
        return self.masks

class BaseVisualEncoder(nn.Module, metaclass=ABCMeta):
    def __init__(self, cfg):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(cfg.PIXEL_MEAN).view(-1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.Tensor(cfg.PIXEL_STD).view(-1, 1, 1), persistent=False)
        self._size_divisibility = cfg.SIZE_DIVISIBILITY
        self._image_size = cfg.IMAGE_SIZE  # int
        self._resize_type = cfg.RESIZE_TYPE  # str
        self._image_scale = cfg.IMAGE_SCALE
        self._test_image_size = cfg.TEST_IMAGE_SIZE  # int or float
        self._test_resize_type = cfg.TEST_RESIZE_TYPE  # str
        self._test_image_scale = cfg.TEST_IMAGE_SCALE
        self._feature_suffix = cfg.FEATURE_SUFFIX  # str
        self._finetune_none = cfg.FINETUNE_TYPE == 'none'
        
        assert self._resize_type in ('none', 'short', 'square', 'rel')
        assert self._test_resize_type in ('none', 'short', 'square', 'rel')
        
        if self._feature_suffix == '_f' and len(cfg.LOAD_FROM) > 0:
            if comm.get_local_rank() == 0:
                download_mask_generator(cfg.LOAD_FROM, _logger)  
            comm.synchronize()   
        
    def _normalize_img(self, x: Tensor) -> Tensor:
        return (x - self.pixel_mean) / self.pixel_std

    @property
    def image_size(self):
        return self._test_image_size if not self.training or self._finetune_none else self._image_size

    @property
    def resize_type(self):
        return self._test_resize_type if not self.training or self._finetune_none else self._resize_type
        
    @property
    def image_scale(self):
        return self._test_image_scale if not self.training or self._finetune_none else self._test_image_scale
    
    def _process_input(self, imgs: List[Tensor], masks: Optional[List[Tensor]]) -> PaddedList:
        processed_imgs = []
        processed_masks = [] if masks is not None else None     
        for i, img in enumerate(imgs):
            if self.resize_type == 'rel':
                processed_imgs.append(F.interpolate(img[None], scale_factor=self.image_scale, mode='bilinear', align_corners=False)[0])
            elif self.resize_type == 'short':
                h, w = img.shape[-2:]
                if h < w:
                    new_h, new_w = self.image_size, int(w * self.image_size / h)
                else:
                    new_h, new_w = int(h * self.image_size / w), self.image_size
                processed_imgs.append(F.interpolate(img[None], size=(new_h, new_w), mode='bilinear', align_corners=False)[0])
            elif self.resize_type == 'square':
                processed_imgs.append(F.interpolate(img[None], size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)[0])
            elif self.resize_type == 'none':
                processed_imgs.append(img.clone())
                
            processed_shape = processed_imgs[-1].shape[-2:]
                
            if masks is not None:
                mask = masks[i]
                if mask.shape[0] == 0:
                    processed_masks.append(mask.new_zeros((0, *processed_shape)))
                else:
                    processed_masks.append(F.interpolate(mask[None], size=processed_shape, mode='bilinear')[0])  
        return PaddedList.from_tensors(processed_imgs, masks=processed_masks, size_divisibility=self.size_divisibility)
        
    
    
    def forward(self, imgs: List[Tensor], masks: Optional[List[Tensor]]=None) -> Dict[str, Tensor]:
        normed_imgs = [self._normalize_img(img) for img in imgs]
        inputs = self._process_input(normed_imgs, masks)
        rtn_dict = self.extract_features(inputs)
        input_key = f"input{self._feature_suffix}"
        assert input_key not in rtn_dict
        rtn_dict[input_key] = inputs
        return rtn_dict
            
            
    @abstractmethod
    def extract_features(self, inputs: PaddedList) -> Dict[str, Tensor]:
        pass
    
    
    @abstractmethod
    def output_shape(self) -> Dict[str, ShapeSpec]:
        pass
    
    @property
    def size_divisibility(self):
        return self._size_divisibility