from typing import List, Tuple, Any, Dict
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable

from .build import SEGMENTOR_REGISTRY
from .base import BaseSegmentor
from .fcclip import MSDeformAttnPixelDecoder, MultiScaleMaskedTransformerDecoder


logger = logging.getLogger(__name__)


@SEGMENTOR_REGISTRY.register()
class FCCLIPSegmentor(BaseSegmentor):
    @configurable
    def __init__(self,
                 dim_latent: int,
                 pixel_decoder: nn.Module,
                 transformer_decoder: nn.Module,
                 pretrained: str,
                 ) -> None:
        super().__init__(mask_is_padded=True, pretrained=pretrained)
        
        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_decoder
        self.void_embedding = nn.Embedding(1, dim_latent)
        
        fcclip_ckpt = torch.load(pretrained, map_location="cpu")
        if "model" in fcclip_ckpt:
            fcclip_ckpt = fcclip_ckpt["model"]
        
        head_dict = {}
        head_str = "sem_seg_head."
        for k, v in fcclip_ckpt.items():
            if k.startswith(head_str):
                head_dict[k[len(head_str):]] = v
                
        head_dict["void_embedding.weight"] = fcclip_ckpt["void_embedding.weight"]
        
        missing_keys, unexpected_keys = self.load_state_dict(head_dict, strict=False)
        if len(missing_keys) > 0:
            logger.warning("Missing keys when loading pretrained FCCLIPSegmentor from %s:\n%s",
                           pretrained, missing_keys)
        if len(unexpected_keys) > 0:
            logger.warning("Unexpected keys when loading pretrained FCCLIPSegmentor from %s:\n%s",
                           pretrained, unexpected_keys)

        for param in self.parameters():
            param.requires_grad = False
            
        logger.info("Loaded pretrained FCCLIPSegmentor from %s", pretrained) 

    @classmethod
    def from_config(cls, cfg, input_shape):
        cls_emb_size = input_shape["emb_f"].channels
        input_shape =  {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.MASKCLIPPP.SEGMENTOR.IN_FEATURES
        }
        
        pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape,
            transformer_dropout=cfg.MODEL.MASK_FORMER.DROPOUT,
            transformer_nheads=cfg.MODEL.MASK_FORMER.NHEADS,
            transformer_dim_feedforward=1024,
            transformer_enc_layers=cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS,
            conv_dim=cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            mask_dim=cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
            norm=cfg.MODEL.SEM_SEG_HEAD.NORM,
            transformer_in_features=cfg.MODEL.MASKCLIPPP.SEGMENTOR.TRANSFORMER_IN_FEATURES,
            common_stride=cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        )
        
        transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        transformer_decoder = MultiScaleMaskedTransformerDecoder(
            in_channels=transformer_predictor_in_channels,
            hidden_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            clip_embedding_dim=cls_emb_size,
            num_queries=cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            nheads=cfg.MODEL.MASK_FORMER.NHEADS,
            dim_feedforward=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            dec_layers=cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1,
            pre_norm=cfg.MODEL.MASK_FORMER.PRE_NORM,
            mask_dim=cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
            enforce_input_project=cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ,
        )
        
        return {
            "dim_latent": cls_emb_size,
            "pixel_decoder": pixel_decoder,
            "transformer_decoder": transformer_decoder,
            "pretrained": cfg.MODEL.MASKCLIPPP.SEGMENTOR.PRETRAINED,
        }
    

    def is_closed_classifier(self) -> bool:
        return False

    def generate_masks(self, batched_inputs: Dict[str, Any], encode_dict: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor] | torch.Tensor | None]:
        t_embs = encode_dict["test_t_embs_f"]
        num_synonyms = encode_dict["num_synonyms"]
        all_embs = torch.cat([t_embs, F.normalize(self.void_embedding.weight, dim=-1)], dim=0)  # Ka+1,D
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(encode_dict)
        predictions = self.predictor(multi_scale_features, mask_features, all_embs, num_synonyms)
        pred_masks = predictions["pred_masks"].sigmoid()
        pred_logits = predictions["pred_logits"]
        return pred_masks, pred_logits
        