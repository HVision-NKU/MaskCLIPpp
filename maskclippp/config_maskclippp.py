# -*- coding: utf-8 -*-
"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/config.py
"""
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75


def add_maskclippp_config(cfg):
    cfg.MODEL.MASKCLIPPP = CN()
    
    def _add_vencoder(cfg, suffix):
        cfg.NAME = "none"
        cfg.MODEL_NAME = ""
        cfg.PRETRAINED = ""
        cfg.LOAD_FROM = ""
        cfg.LOAD_BEG_KEY = ""
        cfg.OUT_FEATURES = []
        cfg.FEATURE_SUFFIX = suffix
        cfg.FINETUNE_TYPE = "none"
        cfg.PIXEL_MEAN = [122.7709383, 116.7460125, 104.09373615]
        cfg.PIXEL_STD = [68.5005327, 66.6321579, 70.32316305]
        cfg.SIZE_DIVISIBILITY = 0
        cfg.IMAGE_SIZE = 640
        cfg.IMAGE_SCALE = 1.0
        cfg.RESIZE_TYPE = "none"
        cfg.TEST_IMAGE_SIZE = 640
        cfg.TEST_IMAGE_SCALE = 1.0        
        cfg.TEST_RESIZE_TYPE = "none"
        
        # for V2
        cfg.MASK_PRIOR_BEG = 0
        cfg.DOWNSAMPLE_METHOD = 'bilinear'
        cfg.DOWN_MASK_THRESH = 0.0
        cfg.MASK_SCALE = 1.0
        cfg.MASK_BIAS = 0.0
        cfg.MASK_LOGIT_SCALE = 0.0
        cfg.LEARNABLE_MASK_LOGIT_SCALE = False
        
    cfg.MODEL.MASKCLIPPP.VISUAL_ENCODER = CN()
    cfg.MODEL.MASKCLIPPP.VISUAL_ENCODER_F = CN()
    
    _add_vencoder(cfg.MODEL.MASKCLIPPP.VISUAL_ENCODER, "")
    _add_vencoder(cfg.MODEL.MASKCLIPPP.VISUAL_ENCODER_F, "_f")
    
    cfg.SOLVER.VISUAL_ENCODER_MULTIPLIER = 0.001
    cfg.SOLVER.TEXT_ENCODER_MULTIPLIER = 0.001
    
    def _add_tencoder(cfg):
        cfg.NAME = "none"
        cfg.MODEL_NAME = ""
        cfg.PRETRAINED = ""
        cfg.LOAD_FROM = ""
        cfg.LOAD_BEG_KEY = ""
        cfg.FINETUNE_TYPE = "none"
        cfg.SKIP_LN_FINAL = False
    
    cfg.MODEL.MASKCLIPPP.TEXT_ENCODER = CN()
    cfg.MODEL.MASKCLIPPP.TEXT_ENCODER_F = CN()
    
    _add_tencoder(cfg.MODEL.MASKCLIPPP.TEXT_ENCODER)
    _add_tencoder(cfg.MODEL.MASKCLIPPP.TEXT_ENCODER_F)
    
    
    cfg.MODEL.MASKCLIPPP.SEGMENTOR = CN()
    cfg.MODEL.MASKCLIPPP.SEGMENTOR.NAME = "FCCLIPSegmentor"
    cfg.MODEL.MASKCLIPPP.SEGMENTOR.PRETRAINED = ""
    cfg.MODEL.MASKCLIPPP.SEGMENTOR.IN_FEATURES = ["stage1_f", "stage2_f", "stage3_f", "stage4_f"]
    cfg.MODEL.MASKCLIPPP.SEGMENTOR.TRANSFORMER_IN_FEATURES = ["stage2_f", "stage3_f", "stage4_f"]
    cfg.MODEL.MASKCLIPPP.SEGMENTOR.OFFLINE_ANN_DIR = ""
    cfg.MODEL.MASKCLIPPP.SEGMENTOR.OFFLINE_ANN_SUFFIX = ""
    
    cfg.MODEL.MASKCLIPPP.USE_LOGIT_SCALE = False
    
    
    cfg.MODEL.MASKCLIPPP.PSM = CN()
    psm_cfg = cfg.MODEL.MASKCLIPPP.PSM
    psm_cfg.NAME = "CorrelationDecoderV1"
    psm_cfg.CORR_WIDTH = 640
    psm_cfg.NUM_HEADS = 8
    psm_cfg.IN_FEATURES = []
    psm_cfg.DETACH_VISUAL_COND = False
    psm_cfg.NORM_VISUAL_COND = False
    psm_cfg.CORR_RESIDUAL = False
    psm_cfg.USE_LOGIT_SCALE = False
    
    psm_cfg.ATTENTION_PROBS_DROPOUT_PROB = 0.0
    psm_cfg.HIDDEN_DROPOUT_PROB = 0.0

    cfg.MODEL.MASKCLIPPP.CRITERION = CN()
    cfg.MODEL.MASKCLIPPP.CRITERION.NAME = "ReweightCELoss"
    cfg.MODEL.MASKCLIPPP.CRITERION.TEMPERATURE = 1.0
    cfg.MODEL.MASKCLIPPP.CRITERION.BALANCE_CLS = False
    cfg.MODEL.MASKCLIPPP.CRITERION.IGNORE_NAN = False
    cfg.MODEL.MASKCLIPPP.CRITERION.IGNORE_EMPTY = False
    cfg.MODEL.MASKCLIPPP.CRITERION.IGNORE_INDEX = -100

    
    cfg.MODEL.MASKCLIPPP.TEST = CN()
    cfg.MODEL.MASKCLIPPP.TEST.ENSEMBLE_ON = False
    cfg.MODEL.MASKCLIPPP.TEST.GEOMETRIC_ENSEMBLE_ALPHA = 1.0
    cfg.MODEL.MASKCLIPPP.TEST.GEOMETRIC_ENSEMBLE_BETA = 1.0
    cfg.MODEL.MASKCLIPPP.TEST.ENSEMBLE_ON_VALID_MASK = 0.0
    
    cfg.MODEL.MASKCLIPPP.TEST.SEGMENTOR_ONLY = False
    cfg.MODEL.MASKCLIPPP.TEST.MASK_ACC = False
    
    cfg.MODEL.MASKCLIPPP.TEMPLATES = "t14"
    cfg.MODEL.MASKCLIPPP.TEMPLATES_F = "t14"
    cfg.MODEL.MASKCLIPPP.TEXT_CHUNK_SIZE = 512
    
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_BOX_ON = False
    
    cfg.RUN_DEMO = False
    
    cfg.WANDB = CN()
    cfg.WANDB.ENABLED = False
    cfg.WANDB.PROJECT = "maskclippp"
    cfg.WANDB.NAME = None