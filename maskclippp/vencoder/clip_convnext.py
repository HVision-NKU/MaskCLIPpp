from typing import Tuple, List, Optional, Dict
import logging
from pathlib import Path
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils import comm
import open_clip
from open_clip.timm_model import TimmModel
from timm.models.convnext import ConvNeXtStage, ConvNeXtBlock

from detectron2.config import configurable
from detectron2.modeling import ShapeSpec

from .build import VISUAL_ENCODER_REGISTRY
from .base import BaseVisualEncoder, PaddedList
from ..utils.misc import downsample_masks
from ..utils.ckpt import load_state_dict_with_beg_key

_logger = logging.getLogger(__name__)

@VISUAL_ENCODER_REGISTRY.register()
class CLIPConvNeXt(BaseVisualEncoder):
    @configurable
    def __init__(self, cfg):
        super().__init__(cfg)
        model_name = cfg.MODEL_NAME
        pretrained = cfg.PRETRAINED
        self._out_features = cfg.OUT_FEATURES
        
        assert 'convnext' in model_name.lower()
        if comm.get_local_rank() == 0:
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        comm.synchronize()
        
        clip_model = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)[0]
        self.proxy: TimmModel = clip_model.visual
        
        if len(cfg.LOAD_FROM) > 0:
            load_from_path = Path(cfg.LOAD_FROM)
            if not load_from_path.exists():
                raise FileNotFoundError(f"LOAD_FROM {load_from_path} does not exist")
            load_ckpt = torch.load(load_from_path, map_location='cpu')
            if 'model' in load_ckpt:
                load_ckpt = load_ckpt['model']
            load_state_dict_with_beg_key(self.proxy, load_ckpt, cfg.LOAD_BEG_KEY, 
                                         self.__class__.__name__+".proxy", load_from_path, _logger)
                
        self._output_strides = {}
        self._output_channels = {}
        for i in range(1, 5):
            self._output_channels[f"stage{i}{self._feature_suffix}"] = self.proxy.trunk.stages[i-1].blocks[0].conv_dw.out_channels
            self._output_strides[f"stage{i}{self._feature_suffix}"] = 2 ** (i + 1)
            if i == 5:
                mi = ''
            else:
                mi = f'-{5 - i}'
            self._output_channels[f"m_embs{mi}{self._feature_suffix}"] = self.proxy.trunk.stages[i-1].blocks[0].conv_dw.out_channels
            self._output_strides[f"m_embs{mi}{self._feature_suffix}"] = -1
            self._output_channels[f"i_embs{i}{self._feature_suffix}"] = self.proxy.trunk.stages[i-1].blocks[0].conv_dw.out_channels
            self._output_strides[f"i_embs{i}{self._feature_suffix}"] = -1
            
            
        self._output_channels[f"m_embs{self._feature_suffix}"] = clip_model.text_projection.shape[-1]
        self._output_strides[f"m_embs{self._feature_suffix}"] = -1
        self._output_channels[f"p_embs{self._feature_suffix}"] = clip_model.text_projection.shape[-1]
        self._output_strides[f"p_embs{self._feature_suffix}"] = 32
        self._output_channels[f"i_embs{self._feature_suffix}"] = clip_model.text_projection.shape[-1]
        self._output_strides[f"i_embs{self._feature_suffix}"] = -1
        self._output_channels[f"emb{self._feature_suffix}"] = clip_model.text_projection.shape[-1]
        self._output_strides[f"emb{self._feature_suffix}"] = -1
        
        self._downsample_method = cfg.DOWNSAMPLE_METHOD
        self._mask_prior_beg = cfg.MASK_PRIOR_BEG
        assert isinstance(self.proxy.trunk.norm_pre, nn.Identity), type(self.proxy.trunk.norm_pre)
        assert len(self.proxy.trunk.stages) == 4
        assert 0 < self._mask_prior_beg <= 5, f"Invalid MASK_PRIOR_BEG {self._mask_prior_beg}, should be in [1, 5]"
                
        self._freeze(cfg.FINETUNE_TYPE)
    
    @classmethod
    def from_config(cls, cfg, all_cfg):
        return {"cfg": cfg}
    
    
    def _freeze(self, finetune_type):
        if finetune_type == 'all':
            return
        elif finetune_type == 'none':
            for p in self.parameters():
                p.requires_grad = False
            return
        elif finetune_type == 'before_proj':
            for p in self.proxy.parameters():
                p.requires_grad = True
            for p in self.proxy.trunk.head.parameters():
                p.requires_grad = False
            for p in self.proxy.head.parameters():
                p.requires_grad = False
        elif finetune_type == 'before_s4':
            for p in self.proxy.parameters():
                p.requires_grad = True
            for p in self.proxy.trunk.stages[3].parameters():
                p.requires_grad = False
            for p in self.proxy.trunk.head.parameters():
                p.requires_grad = False
            for p in self.proxy.head.parameters():
                p.requires_grad = False
        elif finetune_type == 'after_s4':
            for p in self.proxy.parameters():
                p.requires_grad = False
            for p in self.proxy.trunk.stages[3].parameters():
                p.requires_grad = True
            for p in self.proxy.trunk.head.parameters():
                p.requires_grad = True
            for p in self.proxy.head.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown finetune type {finetune_type} for {self.__class__.__name__}")
        
    
    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        return {
            name: ShapeSpec(
                channels=self._output_channels[name], stride=self._output_strides[name]
            )
            for name in self._output_channels.keys()
        }
        
    def extract_features(self, inputs: PaddedList) -> Dict[str, Tensor]:
        if self._finetune_none:
            self.eval()
            with torch.no_grad():
                return self._extract_features(inputs.images, inputs.masks)
        else:
            return self._extract_features(inputs.images, inputs.masks)
        
    def _feat_to_m_embs(self, x: Tensor, masks: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        m_embs = []
        areas = []
        for i, mask in enumerate(masks):
            if mask.shape[0] == 0:
                m_embs.append(x.new_zeros(0, x.shape[1]))
                continue
            down_mask = downsample_masks(mask, x.shape[-2:], self._downsample_method)
            areas.append(down_mask.sum(dim=(1, 2)))
            m_emb = self._mask_pool(x[i], down_mask)
            m_embs.append(m_emb)
        return m_embs, areas
        
    # def _downsample_with_m_embs(self, downsample: nn.Sequential, padded_m_embs: Tensor):
    #     B, Q, C = padded_m_embs.shape
    #     padded_m_embs = padded_m_embs.view(B * Q, C, 1, 1)
    #     padded_m_embs = padded_m_embs.expand(-1, -1, 2, 2)
    #     padded_m_embs = downsample(padded_m_embs)
    #     padded_m_embs = padded_m_embs.view(B, Q, -1)
    #     return padded_m_embs
        
    # def _block_with_m_embs(self, block: ConvNeXtBlock, padded_m_embs: Tensor):
    #     B, Q, C = padded_m_embs.shape
    #     padded_m_embs = padded_m_embs.view(B * Q, C, 1, 1)
    #     padded_m_embs = block(padded_m_embs)
    #     padded_m_embs = padded_m_embs.view(B, Q, C)
    #     return padded_m_embs
        
        
    # def _stage_with_m_embs(self, stage: ConvNeXtStage, m_embs: List[Tensor]) -> List[Tensor]:
    #     num_masks = [m.shape[0] for m in m_embs]
    #     B = len(num_masks)
    #     max_num_masks = max(num_masks)
    #     padded_m_embs = m_embs[0].new_zeros(B, max_num_masks, *m_embs[0].shape[1:])
    #     for i, m_emb in enumerate(m_embs):
    #         padded_m_embs[i, :num_masks[i]] = m_emb
    #     padded_m_embs = self._downsample_with_m_embs(stage.downsample, padded_m_embs)
    #     for block in stage.blocks:
    #         padded_m_embs = self._block_with_m_embs(block, padded_m_embs)
    #     m_embs = []
    #     for i, num in enumerate(num_masks):
    #         m_embs.append(padded_m_embs[i, :num])
    #     return m_embs
    
    
    def _downsample_with_m_embs(self, downsample: nn.Sequential, cated_m_embs: Tensor, x: Tensor, masks: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        N, C = cated_m_embs.shape
        cated_m_embs = cated_m_embs.view(N, C, 1, 1)
        cated_m_embs = cated_m_embs.expand(-1, -1, 2, 2)
        cated_m_embs = downsample(cated_m_embs)
        cated_m_embs = cated_m_embs.view(N, -1)
        x = downsample(x)
        m_embs_from_x, areas = self._feat_to_m_embs(x, masks)
        cated_m_embs += torch.cat(m_embs_from_x, dim=0)
        areas = torch.cat(areas, dim=0)
        return x, cated_m_embs, areas
        
    def _block_with_m_embs(self, block: ConvNeXtBlock, cated_m_embs: Tensor, x: Tensor, masks: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        N, C = cated_m_embs.shape
        cated_m_embs = cated_m_embs.view(N, C, 1, 1)
        cated_m_embs = block(cated_m_embs)
        cated_m_embs = cated_m_embs.view(N, -1)
        x = block(x)
        m_embs_from_x, areas = self._feat_to_m_embs(x, masks)
        cated_m_embs += torch.cat(m_embs_from_x, dim=0)
        areas = torch.cat(areas, dim=0)        
        return x, cated_m_embs, areas
        
    def _stage_with_m_embs(self, stage: ConvNeXtStage, m_embs: Optional[List[Tensor]], areas: Optional[List[bool]],
                           x: Tensor, masks: List[Tensor]) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        if m_embs is None:
            m_embs, areas = self._feat_to_m_embs(x, masks)
        num_masks = [m.shape[0] for m in m_embs]
        cated_m_embs = torch.cat(m_embs, dim=0)
        cated_areas = torch.cat(areas, dim=0)
        x, cated_m_embs, areas_by_downsample = self._downsample_with_m_embs(stage.downsample, cated_m_embs, x, masks)
        cated_areas = torch.maximum(cated_areas, areas_by_downsample)
        for block in stage.blocks:
            x, cated_m_embs, areas_by_block = self._block_with_m_embs(block, cated_m_embs, x, masks)
            cated_areas = torch.maximum(cated_areas, areas_by_block)
        m_embs_rtn = []
        areas_rtn = []
        j = 0
        for num in num_masks:
            m_embs_rtn.append(cated_m_embs[j:j+num])
            areas_rtn.append(cated_areas[j:j+num])
            j += num
        return x, m_embs_rtn, areas_rtn

    # def _add_m_embs(self, m_embs_1: List[Tensor], m_embs_2: List[Tensor]):
    #     m_embs = [m1 + m2 for m1, m2 in zip(m_embs_1, m_embs_2)]
    #     return m_embs
        
    def _extract_features(self, x: Tensor, masks: List[Tensor]):
        out = {}
        x = self.proxy.trunk.stem(x)
        m_embs = None
        areas = None
        for i, stage in enumerate(self.proxy.trunk.stages, 1):
            fk = f"stage{i}{self._feature_suffix}"
            ik = f"i_embs{i}{self._feature_suffix}"
            mk = f"m_embs-{5 - i}{self._feature_suffix}"
            if i >= self._mask_prior_beg or mk in self._out_features:
                x, m_embs, areas = self._stage_with_m_embs(stage, m_embs, areas, x, masks)
            else:
                x = stage(x)
            if fk in self._out_features:
                out[fk] = x.contiguous()
            if ik in self._out_features:
                out[ik] = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            if mk in self._out_features:
                out[mk] = m_embs
            if i < self._mask_prior_beg:
                m_embs = None
                areas = None
            
        # x = self.proxy.trunk.norm_pre(x)  # Identity
        mk = f"m_embs{self._feature_suffix}"
        ek = f"areas{self._feature_suffix}"
        if mk in self._out_features:
            if self._mask_prior_beg == 5:
                m_embs, areas = self._feat_to_m_embs(x, masks)
            proj_m_embs = []
            for m_emb in m_embs:
                m_emb = m_emb.view(*m_emb.shape, 1, 1)
                proj_m_embs.append(self.proxy.head(self.proxy.trunk.head(m_emb)))
            out[mk] = proj_m_embs
            out[ek] = areas
        
        ik = f"i_embs{self._feature_suffix}"
        if ik in self._out_features:
            i_embs = self.proxy.head(self.proxy.trunk.head(x))
            out[ik] = i_embs[:, :, None, None]
        
        pk = f"p_embs{self._feature_suffix}"
        if pk in self._out_features:
            x_shape = x.shape[-2:]
            batch_size = x.shape[0]
            x = x.flatten(2).transpose(1, 2) # B,C,H,W -> B,(HW),C
            x = x.view(-1, x.shape[-1], 1, 1)
            p_embs = self.proxy.head(self.proxy.trunk.head(x))  # B(HW),D
            p_embs = p_embs.view(batch_size, -1, p_embs.shape[-1])
            p_embs = p_embs.transpose(1, 2).view(batch_size, p_embs.shape[-1], *x_shape).contiguous()
            out[pk] = p_embs      
            
        
        return out
            
    def _mask_pool(self, feat: Tensor, mask: Tensor):
        denorm = mask.sum(dim=(1, 2), keepdim=True)  # Q
        m_emb = torch.einsum('chw,qhw->qc', feat, mask / (denorm + 1e-6))
        return m_emb