"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/utils/misc.py

Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
from typing import List, Optional, Tuple, Sequence

import os
import os.path as osp
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
from torch import Tensor
# try:
#     from sklearnex import patch_sklearn
#     patch_sklearn()
# except ImportError:
#     print("pip install scikit-learn-intelex")
from sklearn.cluster import KMeans

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def read_file_list(split_path, default_dir, default_suffix):
    file_list = []
    if split_path is not None:
        with open(split_path, 'r') as fp:
            while True:
                a_line = fp.readline()
                if not a_line:
                    break
                file_list.append(a_line.strip())
    else:
        for fname in os.listdir(default_dir):
            if fname.endswith(default_suffix):
                name = osp.splitext(fname)[0]
                file_list.append(name)
    return file_list

def downsample_masks(masks: Tensor, target_shape: Tuple[int, int], method: str):
    """_summary_

    Args:
        masks (Tensor): Q,H,W, float
        target_shape (Tuple[int, int]): _description_
        method (str): _description_
    """
    if method in ('nearest', 'bilinear', 'bicubic'):
        downed = F.interpolate(masks.unsqueeze(0), size=target_shape, mode=method, align_corners=False).squeeze(0)
    elif method == 'avg':
        downed = F.adaptive_avg_pool2d(masks.unsqueeze(0), target_shape).squeeze(0)
    elif method == 'max':
        downed = F.adaptive_max_pool2d(masks.unsqueeze(0), target_shape).squeeze(0)
    return downed


def split_mask(mask: np.ndarray, num_clusters: int) -> np.ndarray:
    """Split a binary mask into multiple masks using KMeans clustering.

    Args:
        mask (np.ndarray): Binary mask
        num_clusters (int): number of clusters to split the mask into

    Returns:
        np.ndarray: Binary masks
    """
    assert num_clusters > 0
    if num_clusters == 1:
        return np.array([mask])
    coords = np.column_stack(np.where(mask > 0))
    
    kmeans = KMeans(n_clusters=num_clusters).fit(coords)
    labels = kmeans.labels_
    
    sub_masks = []
    for i in range(num_clusters):
        sub_mask = np.zeros_like(mask)
        sub_mask[coords[labels == i, 0], coords[labels == i, 1]] = 255
        sub_masks.append(sub_mask)
    
    return np.stack(sub_masks, axis=0)


def merge_masks(masks: Sequence[np.ndarray]) -> np.ndarray:
    """Merge multiple binary masks into a single mask.

    Args:
        masks (Sequence[np.ndarray]): Binary masks

    Returns:
        np.ndarray: Merged binary mask
    """
    mask_shape = masks[0].shape
    flatted_masks = [m.flatten() for m in masks]
    merge_masks = np.any(flatted_masks, axis=0).reshape(mask_shape)
    return merge_masks

def dilate_mask(mask: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    """Dilate a uint8 mask.

    Args:
        mask (np.ndarray): uint8 mask
        kernel_size (int): kernel size for dilation
        iterations (int): iterations for dilation

    Returns:
        np.ndarray: Dilated uint8 mask
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated_mask


def mask_iou(masks1: np.ndarray, masks2: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Calculate the IoU between two sets of masks.

    Args:
        masks1 (np.ndarray): shape (N, H, W) binary arrays
        masks2 (np.ndarray): shape (M, H, W) binary arrays
        scale (float): scale factor for masks
    
    Returns:
        np.ndarray: shape (N, M) IoU matrix
    """
    assert masks1.shape[1:] == masks2.shape[1:]
    # resize masks
    # dsize=(int(masks1.shape[2] * scale), int(masks1.shape[1] * scale))
    resized_masks1 = F.interpolate(torch.from_numpy(masks1).unsqueeze(0).float(), scale_factor=scale, mode='nearest').squeeze(0).numpy().astype(np.bool_)
    resized_masks2 = F.interpolate(torch.from_numpy(masks2).unsqueeze(0).float(), scale_factor=scale, mode='nearest').squeeze(0).numpy().astype(np.bool_)
    
    N = resized_masks1.shape[0]
    M = resized_masks2.shape[0]
    masks1_flat = resized_masks1.reshape(N, -1)
    masks2_flat = resized_masks2.reshape(M, -1)

    intersection = np.sum(np.logical_and(masks1_flat[:, None, :], masks2_flat[None, :, :]), axis=-1)
    union = np.sum(np.logical_or(masks1_flat[:, None, :], masks2_flat[None, :, :]), axis=-1)
    
    iou = intersection / (union + 1e-6)
    return iou


def mask_iou_torch(masks1: Tensor, masks2: Tensor, scale: float = 1.0) -> Tensor:
    """
    Calculate the IoU between two sets of masks.

    Args:
        masks1 (Tensor): shape (N, H, W) binary arrays
        masks2 (Tensor): shape (M, H, W) binary arrays
    
    Returns:
        Tensor: shape (N, M) IoU matrix
    """
    assert masks1.shape[1:] == masks2.shape[1:]
    # resize masks
    resized_masks1 = F.interpolate(masks1.float().unsqueeze(0), scale_factor=scale, mode='nearest').squeeze(0).bool()
    resized_masks2 = F.interpolate(masks2.float().unsqueeze(0), scale_factor=scale, mode='nearest').squeeze(0).bool()
    
    N = resized_masks1.shape[0]
    M = resized_masks2.shape[0]
    masks1_flat = resized_masks1.reshape(N, -1)
    masks2_flat = resized_masks2.reshape(M, -1)

    intersection = torch.sum(masks1_flat[:, None, :] & masks2_flat[None, :, :], dim=-1)
    union = torch.sum(masks1_flat[:, None, :] | masks2_flat[None, :, :], dim=-1)
    
    iou = intersection / (union + 1e-6)
    return iou


def unseen_id2mask(metadata) -> np.ndarray:
    unseen_ids = metadata.unseen_ids
    if hasattr(metadata, "stuff_dataset_id_to_contiguous_id"):
        stuff_dataset_id_to_contiguous_id = metadata.stuff_dataset_id_to_contiguous_id
        contigious_unseen_ids = [stuff_dataset_id_to_contiguous_id[unseen_id] for unseen_id in unseen_ids]
    else:
        contigious_unseen_ids = unseen_ids
    unseen_mask = np.zeros(len(stuff_dataset_id_to_contiguous_id), dtype=np.bool)
    unseen_mask[contigious_unseen_ids] = True
    return unseen_mask
    

TEMPLATES = {
    "t1": [
        "a photo of a {}.",
    ],
    "t14": [  # from fc-clip
        "a photo of a {}.",
        "This is a photo of a {}",
        "There is a {} in the scene",
        "There is the {} in the scene",
        "a photo of a {} in the scene",
        "a photo of a small {}.",
        "a photo of a medium {}.",
        "a photo of a large {}.",
        "This is a photo of a small {}.",
        "This is a photo of a medium {}.",
        "This is a photo of a large {}.",
        "There is a small {} in the scene.",
        "There is a medium {} in the scene.",
        "There is a large {} in the scene.",
    ],
    "t80": [  # from sed
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
    ]
}