from typing import Tuple
import numpy as np
import cv2
import torch
import torch.nn.functional as F

def cal_affinity(feat: torch.Tensor,
                 method: str = 'cos'):
    """Calculate Affnity

    Args:
        feat (torch.Tensor): BCHW

    Returns:
        _type_: _description_
    """
    fshape = feat.shape[-2:]
    feat = feat.reshape(*feat.shape[:2], -1)  # B,C,L
    if method == 'cos':
        feat = F.normalize(feat, dim=1)
        aff = torch.bmm(feat.transpose(1, 2), feat) # B,L,L
        # B,L,C = feat_t.shape
        # feat_t = feat_t.view(-1, C)
        # aff = F.cosine_similarity(feat_t.unsqueeze(1), feat_t.unsqueeze(0), dim=-1)
        # aff = aff.view(B, L, L)
    elif method == 'l2':
        feat_t = feat.transpose(1, 2)  # B,L,C
        aff = torch.cdist(feat_t, feat_t, p=2)
    aff = aff.reshape(aff.shape[0], fshape[0], fshape[1], fshape[0], fshape[1])
    return aff


def show_affinity_with_image(img: np.ndarray, 
                             affinity: np.ndarray, 
                             pos: Tuple[int, int],
                             alpha: float = 0.3):
    """_summary_

    Args:
        img (np.ndarray): HW3, uint8
        affinity (np.ndarray): (hw)(hw)
        pos (Tuple[int, int]): x,y
    """
    img_shape = img.shape[:2]
    aff_map = affinity[pos[::-1]]
    aff_shape = aff_map.shape
    aff_map = np.uint8(aff_map * 255)
    aff_map = cv2.resize(aff_map, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
    pos_xy  = (int(pos[0] * img_shape[1] / aff_shape[1]), int(pos[1] * img_shape[0] / aff_shape[0]))
    bbox_color = (0, 255, 0)
    aff_heat_map = cv2.applyColorMap(aff_map, cv2.COLORMAP_JET)
    aff_heat_map = cv2.cvtColor(aff_heat_map, cv2.COLOR_BGR2RGB)
    im_show = cv2.addWeighted(img, alpha, aff_heat_map, (1 - alpha), 0)
    im_show = np.uint8(im_show)
    
    bbox_hw = (int(img_shape[0] / aff_shape[0]), int(img_shape[1] / aff_shape[1]))
    cv2.rectangle(im_show, (pos_xy[0], pos_xy[1]), (pos_xy[0] + bbox_hw[1], pos_xy[1] + bbox_hw[0]),
                  bbox_color, 2)
    return im_show