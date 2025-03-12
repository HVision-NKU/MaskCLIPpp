from typing import Tuple, Dict, Optional, List

import logging
import random
import torch
from torch import nn, Tensor
from torch.nn import functional as F

import os
import numpy as np
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
import detectron2.utils.comm as comm

from .vencoder import build_visual_encoder, BaseVisualEncoder, PaddedList
from .tencoder import build_text_encoder
from .segmentor import build_segmentor, BaseSegmentor
from .psm import build_psm
from .criterion import build_criterion, BaseCriterion
from .utils.misc import downsample_masks, TEMPLATES


_logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class MaskCLIPpp(nn.Module):
    __version__ = 2
    
    @configurable
    def __init__(
        self,
        *,
        visual_encoder: BaseVisualEncoder,
        visual_encoder_f: BaseVisualEncoder,
        text_encoder: nn.Module,
        text_encoder_f: nn.Module,
        segmentor: BaseSegmentor,
        psm: nn.Module,
        criterion: BaseCriterion,
        train_metadata,
        test_metadata,
        templates: List[str],
        templates_f: List[str],
        text_chunk_size: int,
        # inference
        sem_seg_postprocess_before_inference: bool,
        overlap_threshold: float,
        object_mask_threshold: float,
        use_logit_scale: bool,
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        instance_box_on: bool,
        mask_acc_on: bool,
        test_topk_per_image: int,
        ensemble_on: bool,
        geometric_ensemble_alpha: float,
        geometric_ensemble_beta: float,
        segmentor_only: bool,
        run_demo: bool
    ):
        super().__init__()
        
        self.visual_encoder = visual_encoder
        self.visual_encoder_f = visual_encoder_f
        
        self.segmentor = segmentor
        self.psm = psm
        self.criterion = criterion
        
        if text_encoder_f is not None:
            assert text_encoder_f.finetune_none
        if not text_encoder.finetune_none:
            self.text_encoder = text_encoder
            
        self.run_demo = run_demo
        if run_demo:
            self.text_encoder = text_encoder
            self.text_encoder_f = text_encoder_f
            
        else:
            if use_logit_scale:
                self.register_parameter("_logit_scale", text_encoder.logit_scale)
            else:
                self.register_buffer("_logit_scale", text_encoder.logit_scale.data, persistent=False)
            
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        self.text_chunk_size = text_chunk_size
        self.use_logit_scale = use_logit_scale
        # inference
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.semantic_on = semantic_on
        self.panoptic_on = panoptic_on
        self.instance_on = instance_on
        self.instance_box_on = instance_box_on
        self.mask_acc_on = mask_acc_on
        self.test_topk_per_image = test_topk_per_image
        
        # ensemble
        self.ensemble_on = ensemble_on
        self.geometric_ensemble_alpha = geometric_ensemble_alpha
        self.geometric_ensemble_beta = geometric_ensemble_beta
        
        self.segmentor_only = segmentor_only
        
        _, train_synonyms = self._prepare_class_names_from_metadata(train_metadata, train_metadata, is_train=True)
        category_overlapping_mask, test_synonyms = self._prepare_class_names_from_metadata(test_metadata, train_metadata, is_train=False)
        self.register_buffer("category_overlapping_mask", category_overlapping_mask, persistent=False)
        
        if self.segmentor.is_closed_classifier():
            self.train_synonyms = train_synonyms
            self.register_buffer("train2test_lut", self._get_train2test_lut(train_synonyms, test_synonyms, category_overlapping_mask), persistent=False)
        
        self.templates = templates
        train_sentences, train_num_synonyms = self._words_to_sentences(train_synonyms, templates)
        test_sentences, test_num_synonyms = self._words_to_sentences(test_synonyms, templates)
        self.register_buffer("train_num_synonyms", train_num_synonyms, persistent=False)
        self.register_buffer("test_num_synonyms", test_num_synonyms, persistent=False)
        
        if not text_encoder.finetune_none:
            self.train_sentences = train_sentences
            self.test_sentences = test_sentences
        else:
            print(f"Begin to prepare text classifiers: train[{len(train_sentences)}] test[{len(test_sentences)}]")
            local_rank = comm.get_local_rank()
            text_encoder = text_encoder.to(device="cuda:%d" % local_rank)
            
            train_t_embs = self._cal_text_emb(text_encoder, train_sentences, distributed=True)
            test_t_embs = self._cal_text_emb(text_encoder, test_sentences, distributed=True)
            train_t_embs = train_t_embs.reshape(sum(train_num_synonyms), len(templates), train_t_embs.size(-1))  # Ka,T,D
            test_t_embs = test_t_embs.reshape(sum(test_num_synonyms), len(templates), test_t_embs.size(-1))  # Ka,T,D
            train_t_embs /= train_t_embs.norm(dim=-1, keepdim=True)
            test_t_embs /= test_t_embs.norm(dim=-1, keepdim=True)
            self.register_buffer("train_t_embs", train_t_embs, persistent=False)
            self.register_buffer("test_t_embs", test_t_embs, persistent=False)
            
        if text_encoder_f is not None:
            local_rank = comm.get_local_rank()
            text_encoder_f = text_encoder_f.to(device="cuda:%d" % local_rank)
            # following are refer to fc-clip
            self.templates_f = templates_f
            test_sentences_f, _ = self._words_to_sentences(test_synonyms, templates_f)
            print(f"Begin to prepare text classifiers for templates_f: test[{len(test_sentences_f)}]")
            test_t_embs_f = self._cal_text_emb(text_encoder_f, test_sentences_f, distributed=True)
            test_t_embs_f /= test_t_embs_f.norm(dim=-1, keepdim=True)
            test_t_embs_f = test_t_embs_f.reshape(sum(test_num_synonyms), len(templates_f), test_t_embs_f.size(-1)).mean(1)  # Ka,D
            test_t_embs_f /= test_t_embs_f.norm(dim=-1, keepdim=True)
            self.register_buffer("test_t_embs_f", test_t_embs_f, persistent=False)
        
                
    @classmethod
    def from_config(cls, cfg):
        visual_encoder = build_visual_encoder(cfg.MODEL.MASKCLIPPP.VISUAL_ENCODER, cfg)
        text_encoder = build_text_encoder(cfg.MODEL.MASKCLIPPP.TEXT_ENCODER)
        visual_encoder_f = build_visual_encoder(cfg.MODEL.MASKCLIPPP.VISUAL_ENCODER_F, cfg)
        text_encoder_f = build_text_encoder(cfg.MODEL.MASKCLIPPP.TEXT_ENCODER_F)
        
        output_shape = visual_encoder.output_shape()
        if visual_encoder_f is not None:
            output_shape.update(visual_encoder_f.output_shape())
        segmentor = build_segmentor(cfg, input_shape=output_shape)
        
        templates = TEMPLATES[cfg.MODEL.MASKCLIPPP.TEMPLATES]
        templates_f = TEMPLATES.get(cfg.MODEL.MASKCLIPPP.TEMPLATES_F, None)

        template_dim = len(templates)
        
        psm = build_psm(cfg.MODEL.MASKCLIPPP.PSM, output_shape, template_dim)
        
        criterion = build_criterion(cfg.MODEL.MASKCLIPPP.CRITERION)
        
        return {
            "visual_encoder": visual_encoder,
            "text_encoder": text_encoder,
            "text_encoder_f": text_encoder_f,
            "visual_encoder_f": visual_encoder_f,
            "segmentor": segmentor,
            "psm": psm,
            "criterion": criterion,
            "train_metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "test_metadata": MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            "templates": templates,
            "templates_f": templates_f,
            "text_chunk_size": cfg.MODEL.MASKCLIPPP.TEXT_CHUNK_SIZE,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "use_logit_scale": cfg.MODEL.MASKCLIPPP.USE_LOGIT_SCALE,
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "instance_box_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_BOX_ON,
            "mask_acc_on": cfg.MODEL.MASKCLIPPP.TEST.MASK_ACC,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "ensemble_on": cfg.MODEL.MASKCLIPPP.TEST.ENSEMBLE_ON,
            "geometric_ensemble_alpha": cfg.MODEL.MASKCLIPPP.TEST.GEOMETRIC_ENSEMBLE_ALPHA,
            "geometric_ensemble_beta": cfg.MODEL.MASKCLIPPP.TEST.GEOMETRIC_ENSEMBLE_BETA,
            "segmentor_only": cfg.MODEL.MASKCLIPPP.TEST.SEGMENTOR_ONLY,
            "run_demo": cfg.RUN_DEMO
        }
    
    @torch.inference_mode()
    def set_metadata(self, test_metadata):
        # test metadata setter for demo
        self.test_metadata = test_metadata
        category_overlapping_mask, test_synonyms = self._prepare_class_names_from_metadata(test_metadata, self.train_metadata, is_train=False)
        self.category_overlapping_mask = category_overlapping_mask.to(self.device)
        if self.segmentor.is_closed_classifier():
            self.train2test_lut = self._get_train2test_lut(self.train_synonyms, test_synonyms, category_overlapping_mask)
        test_sentences, test_num_synonyms = self._words_to_sentences(test_synonyms, self.templates)
        self.test_num_synonyms = test_num_synonyms
        test_t_embs = self._cal_text_emb(self.text_encoder, test_sentences)
        test_t_embs = test_t_embs.reshape(sum(test_num_synonyms), len(self.templates), test_t_embs.size(-1))  # Ka,T,D
        test_t_embs /= test_t_embs.norm(dim=-1, keepdim=True)
        self.test_t_embs = test_t_embs
        
        if hasattr(self, "test_t_embs_f"):
            test_sentences_f, _ = self._words_to_sentences(test_synonyms, self.templates_f)
            test_t_embs_f = self._cal_text_emb(self.text_encoder_f, test_sentences_f)
            test_t_embs_f /= test_t_embs_f.norm(dim=-1, keepdim=True)
            test_t_embs_f = test_t_embs_f.reshape(sum(test_num_synonyms), len(self.templates_f), test_t_embs_f.size(-1)).mean(1)  # Ka,D
            test_t_embs_f /= test_t_embs_f.norm(dim=-1, keepdim=True)
            self.test_t_embs_f = test_t_embs_f
        
    def saved_modules(self):
        saved = {
            "visual_encoder": self.visual_encoder,
            "psm": self.psm,
            "criterion": self.criterion,
        }
        if hasattr(self, "text_encoder"):
            saved["text_encoder"] = self.text_encoder
        return saved
    
    @property
    def logit_scale(self):
        if hasattr(self, "text_encoder"):
            return self.text_encoder.logit_scale
        else:
            return self._logit_scale
    
    def _prepare_extra_synonyms_from_file(self, extra_categories: str, train_synonyms: List[List[str]]) -> List[List[str]]:
        if extra_categories == "":
            return []
        assert os.path.isfile(extra_categories), f"Extra categories file {extra_categories} does not exist."
        with open(extra_categories, 'r') as fp:
            extra_synonyms = fp.read().splitlines()
        extra_synonyms = [x[x.find(':')+1:] for x in extra_synonyms]
        extra_synonyms = [x.split(',') for x in extra_synonyms]
        
        non_overlapped_extra_synonyms = []
        train_class_names = {name for synonyms in train_synonyms for name in synonyms}
        for synonyms in extra_synonyms:
            if set(synonyms).isdisjoint(train_class_names):
                non_overlapped_extra_synonyms.append(synonyms)
        return non_overlapped_extra_synonyms
                
        
    def _prepare_class_names_from_metadata(self, metadata, train_metadata, is_train) -> Tuple[Tensor, List[List[str]]]:
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        # get text classifier
        try:
            class_names = split_labels(metadata.stuff_classes) # it includes both thing and stuff
            train_class_names = split_labels(train_metadata.stuff_classes)
        except:
            # this could be for insseg, where only thing_classes are available
            class_names = split_labels(metadata.thing_classes)
            train_class_names = split_labels(train_metadata.thing_classes)
        
        if is_train:
            category_overlapping_mask = None
        else:
            train_class_names = {l for label in train_class_names for l in label}
            category_overlapping_list = []
            for test_class_names in class_names:
                is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names))
                category_overlapping_list.append(is_overlapping)
            category_overlapping_mask = torch.tensor(
                category_overlapping_list, dtype=torch.bool)
        return category_overlapping_mask, class_names  # K, List(K) of synonyms list
        
        
    def _get_train2test_lut(self, train_synonyms: List[List[str]], test_synonyms: List[List[str]], category_overlapping_mask: Tensor) -> Tensor:
        K_train = len(train_synonyms)
        K_test = len(test_synonyms)
        lut = torch.full((K_train + 1, ), K_test, dtype=torch.long)
        lut_found = torch.zeros(K_train, dtype=torch.bool)
        assert len(test_synonyms) == category_overlapping_mask.size(0)
        for i, one_test_synonyms in enumerate(test_synonyms):
            if category_overlapping_mask[i]:
                for j, one_train_synonyms in enumerate(train_synonyms):
                    if not lut_found[j] and not set(one_train_synonyms).isdisjoint(one_test_synonyms):
                        lut[j] = i
                        lut_found[j] = True
        
        return lut
        
        
    def _words_to_sentences(self, synonyms_list, templates):
        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in templates:
                    res.append(template.format(x))
            return res, len(res) // len(templates)
        
        num_synonyms = []
        templated_class_sentences = []
        for x in synonyms_list:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_sentences += templated_classes
            num_synonyms.append(templated_classes_num)
        num_synonyms = torch.tensor(num_synonyms, dtype=torch.long)
        return templated_class_sentences, num_synonyms
    
    @property
    def device(self):
        return self.logit_scale.device
    
    def _sample_one_synonym(self, t_embs: Tensor, num_of_each_cls: Tensor) -> Tuple[Tensor, Tensor]:
        """_summary_

        Args:
            t_embs (Tensor): Ka,T,D
            num_of_each_cls (Tensor): K

        Returns:
            Tuple[Tensor, Tensor]: K,T,D, K
        """
        K = len(num_of_each_cls)
        t_embs_r = t_embs.new_zeros((K, t_embs.size(1), t_embs.size(2)))
        beg = 0
        for k, num in enumerate(num_of_each_cls):
            if num > 1:
                idx = random.randrange(beg, beg+num)
                t_embs_r[k] = t_embs[idx]
            else:
                t_embs_r[k] = t_embs[beg]
            beg += num
        num_of_each_cls_r = torch.ones_like(num_of_each_cls)
        return t_embs_r, num_of_each_cls_r
    
    def _sample_one_synonym_from_sentences(self, sentences: List[str], num_of_each_cls: Tensor) -> Tuple[List[str], Tensor]:
        num_templates = len(sentences) // sum(num_of_each_cls)
        sentences_r = []
        beg = 0
        for k, num in enumerate(num_of_each_cls):
            if num > 1:
                idx = random.randrange(beg, beg+num)
                sentences_r.extend(sentences[idx*num_templates: (idx+1)*num_templates])
            else:
                sentences_r.extend(sentences[beg*num_templates: (beg+1)*num_templates])
            beg += num
        return sentences_r, torch.ones_like(num_of_each_cls)
    
    def _cal_text_emb(self, text_encoder: nn.Module, class_sentences: List[str], distributed=False) -> Tensor:
        world_size = comm.get_world_size()
        if distributed and world_size > 1:
            local_rank = comm.get_local_rank()
            beg = local_rank * len(class_sentences) // world_size
            end = (local_rank + 1) * len(class_sentences) // world_size
            if local_rank == world_size - 1:
                end = len(class_sentences)
            # print(f"local_rank[{local_rank}] beg[{beg}] end[{end}]")
        else:
            beg = 0
            end = len(class_sentences)
        text_chunk_size = self.text_chunk_size
        t_embs = []
        for i in range(beg, end, text_chunk_size):
            t_embs.append(text_encoder(class_sentences[i: min(i+text_chunk_size, end)]))
        t_embs = torch.cat(t_embs, dim=0)
        if distributed and world_size > 1:
            device = t_embs.device
            t_embs_list = comm.all_gather(t_embs)
            t_embs_list = [t_embs.to(device) for t_embs in t_embs_list]
            t_embs = torch.cat(t_embs_list, dim=0)
        return t_embs
                
    def _is_valid_masks(self,
                        masks: Tensor) -> Tensor:
        """_summary_

        Args:
            masks (Tensor): Q,H,W

        Returns:
            Tensor: Q
        """
        is_valid = masks.sum(dim=[1, 2]) > 0
        return is_valid
    
    def _prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            if hasattr(targets_per_image, "gt_classes"):
                labels = targets_per_image.gt_classes
                if hasattr(self, "contigious_seen_ids_to_contigious_train_ids"):
                    labels = self.contigious_seen_ids_to_contigious_train_ids[labels]
            else:
                labels = targets_per_image.gt_qualities
                if hasattr(self, "contigious_seen_ids_to_contigious_train_ids"):
                    raise NotImplementedError()
            new_targets.append(dict(labels=labels, masks=targets_per_image.gt_masks))
        return new_targets
    
    def _get_pred_logits_from_corr(self,
                                   corrs: Tensor,
                                   mask2batch: Tensor,
                                   valid_masks_by_imgs: List[Tensor],
                                   num_synonyms: Tensor) -> List[Tensor]:
        beg_of_each_cls = torch.cumsum(num_synonyms, dim=0)
        beg_of_each_cls = torch.cat([torch.zeros(1, device=beg_of_each_cls.device, dtype=beg_of_each_cls.dtype), 
                                        beg_of_each_cls])
        B = len(valid_masks_by_imgs)
        K = len(beg_of_each_cls) - 1
        Ka = num_synonyms.sum()
        pred_logits = []
        for b in range(B):
            midx = mask2batch == b
            masks = valid_masks_by_imgs[b]
            corr_per_img = corrs[midx]  # Q, Ka
            Q = masks.size(0)
            if Q == 0:
                pred_logits.append(torch.zeros(0, K, device=self.device, dtype=corrs.dtype))
            else:
                pred_logits_by_cls = []
                for cid in range(K):
                    pred_logits_by_cls.append(
                        corr_per_img[:, beg_of_each_cls[cid]: beg_of_each_cls[cid+1]].max(dim=1).values)
                if corr_per_img.size(1) == Ka + 1:
                    pred_logits_by_cls.append(corr_per_img[:, -1])
                pred_logits_by_cls = torch.stack(pred_logits_by_cls, dim=1)  # Q,K
                pred_logits.append(pred_logits_by_cls)
        return pred_logits
    
        
    def _ensemble_logits(self, 
                         logits: List[Tensor], 
                         logits_f: List[Tensor]) -> List[Tensor]:
        
        if self.segmentor.is_closed_classifier():
            converted_logits_f = []
            lut = self.train2test_lut.unsqueeze(0)
            for logit, logit_f in zip(logits, logits_f):
                converted_logit_f = logit.new_zeros(logit.size(0), logit.size(1) + 1)
                converted_logit_f.scatter_reduce_(1, lut.expand(logit.size(0), -1), logit_f, reduce="mean", include_self=False)
                converted_logits_f.append(converted_logit_f)
        else:
            assert logits_f[0].size(-1) == logits[0].size(-1) + 1
            converted_logits_f = logits_f
        ensembled_logits = []
        category_overlapping_mask = self.category_overlapping_mask.to(torch.long)
        for logit, logit_f in zip(logits, converted_logits_f):
            alpha = self.geometric_ensemble_alpha
            beta = self.geometric_ensemble_beta
            prob = logit.softmax(dim=-1)
            prob_f = logit_f[:, :-1].softmax(dim=-1)  # remove void
            cls_logits_seen = (prob_f ** (1 - alpha) * prob ** alpha + 1e-8).log() * category_overlapping_mask
            cls_logits_unseen = (prob_f ** (1 - beta) * prob ** beta + 1e-8).log() * (1 - category_overlapping_mask)
            ensembled_logits.append(cls_logits_seen + cls_logits_unseen)
        return ensembled_logits
    
    
    def _add_void_logits(self, mask_cls_results: List[Tensor], segmentor_logits: List[Tensor]) -> List[Tensor]:
        """_summary_

        Args:
            mask_cls_results (List[Tensor]): List(B) of Q,K
            segmentor_logits (List[Tensor]): List(B) of Q,K+1
        
        Returns:
            List[Tensor]: List(B) of Q,K+1
        """
        rtn_logits = []
        for cls_result, seg_logit in zip(mask_cls_results, segmentor_logits):
            void_prob = F.softmax(seg_logit, dim=-1)[:, -1:]
            cls_prob = F.softmax(cls_result, dim=-1)
            rtn_prob = torch.cat([cls_prob * (1 - void_prob), void_prob], dim=-1)
            rtn_logits.append(torch.log(rtn_prob + 1e-8))
        return rtn_logits
        
            
    def _get_text_classifiers(self):
        if self.training:
            if hasattr(self, "text_encoder"):
                if hasattr(self, "test_t_embs"):
                    del self.test_t_embs  # text encoder is updated, the test_t_embs is invalid
                sentences, num_synonyms = self._sample_one_synonym_from_sentences(self.train_sentences, self.train_num_synonyms)
                t_embs = self._cal_text_emb(self.text_encoder, sentences)
                t_embs = t_embs.reshape(sum(num_synonyms), -1, t_embs.size(-1))  # K,T,D
                t_embs = t_embs / t_embs.norm(dim=-1, keepdim=True)
            else:
                t_embs, num_synonyms = self.train_t_embs, self.train_num_synonyms
                t_embs, num_synonyms = self._sample_one_synonym(t_embs, num_synonyms)  # K,T,D; K
            
        else:
            if hasattr(self, "text_encoder"):
                num_synonyms = self.test_num_synonyms
                if hasattr(self, "test_t_embs"):
                    t_embs = self.test_t_embs
                else:
                    t_embs = self._cal_text_emb(self.text_encoder, self.test_sentences)
                    t_embs = t_embs.reshape(sum(num_synonyms), -1, t_embs.size(-1))  # K,T,D
                    t_embs = t_embs / t_embs.norm(dim=-1, keepdim=True)
                    self.register_buffer("test_t_embs", t_embs, persistent=False)
            else:
                t_embs, num_synonyms = self.test_t_embs, self.test_num_synonyms

        return t_embs, num_synonyms
    
    def _get_masks(self, batched_inputs, encode_dict, num_synonyms):
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self._prepare_targets(gt_instances)
            masks_by_imgs = [t["masks"].float() for t in targets]
            labels_by_imgs = [t["labels"] for t in targets]
        else:
            if hasattr(self, "test_t_embs_f"):
                encode_dict["test_t_embs_f"] = self.test_t_embs_f
                encode_dict["num_synonyms"] = num_synonyms
            segmentor_logits = self.segmentor(batched_inputs, encode_dict)
            input_f: PaddedList = encode_dict["input_f"]
            masks_by_imgs = input_f.get_unpadded_masks()
        
        if self.training:
            logits_by_imgs = None
        else:
            labels_by_imgs = None
            if segmentor_logits is not None:
                logits_by_imgs = segmentor_logits
            else:
                logits_by_imgs = None
        return masks_by_imgs, labels_by_imgs, logits_by_imgs
    
    def _get_corrs(self, encode_dict, valid_masks_by_imgs, mask2batch, t_embs):
        if mask2batch.size(0) == 0:
            _logger.warning("Number of valid masks in this local-batch is 0")
            corrs = torch.zeros((0, t_embs.size(0), t_embs.size(1)), device=self.device, dtype=t_embs.dtype)
        else:
            m_embs = encode_dict["m_embs"]  # List(B) of Q,D
            m_embs = torch.cat(m_embs, dim=0)  # N,D
            corrs = torch.einsum("ktd,nd->nkt", t_embs, F.normalize(m_embs, dim=-1))  # t_embs is normed when init
            if self.use_logit_scale:
                corrs *= torch.clamp(self.logit_scale.exp(), max=100)

            masks = encode_dict["input"].set_unpadded_masks(valid_masks_by_imgs)  # padded masks for visual_encoder
            masks = torch.cat(masks)  # N
            encode_dict["t_embs"] = t_embs
            corrs = self.psm(corrs, masks, mask2batch, encode_dict)  # N,Ka
        return corrs
    
    
    def _get_loss(self, corrs, t_embs, valid_labels_by_imgs, encode_dict):
        labels = torch.cat(valid_labels_by_imgs)  # N
        
        pred_dict = dict(corrs=corrs)
        pred_dict.update(encode_dict)
        if labels.dim() == 1:
            target_dict = dict(gt_labels=labels)
        else:
            target_dict = dict(gt_soft_labels=labels)
        return self.criterion(pred_dict, target_dict)

        
    def _segmentor_inference(self, batched_inputs):
        assert not self.training, "segmentor_inference should be called in inference mode"
        images = [x["image"].to(self.device) for x in batched_inputs]
        encode_dict = self.visual_encoder_f(images)
        t_embs, num_synonyms = self._get_text_classifiers()
        masks_by_imgs, _, _ = self._get_masks(batched_inputs, encode_dict, num_synonyms)
        results = []
        for pred_masks, input_per_image in zip(masks_by_imgs, batched_inputs):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            resized_pred_masks = retry_if_cuda_oom(downsample_masks)(pred_masks, (height, width), 'bilinear')
            results.append({"pred_masks": resized_pred_masks > 0.5})
        return results
    
    
    def forward(self, batched_inputs):
        if self.segmentor_only:
            return self._segmentor_inference(batched_inputs)
        
        images = [x["image"].to(self.device) for x in batched_inputs]

        # Get visual features for segmentor
        encode_dict = {}
        if self.visual_encoder_f is not None:
            encode_dict.update(self.visual_encoder_f(images))
            
        t_embs, num_synonyms = self._get_text_classifiers()
        
        # Get masks. valid_logits_by_imgs is close-vocabulary (on train set) logits
        valid_masks_by_imgs, valid_labels_by_imgs, valid_logits_by_imgs = self._get_masks(batched_inputs, encode_dict, num_synonyms)
        
        mask2batch = torch.cat([torch.full((mask.shape[0],), bid, device=self.device, dtype=torch.long) for bid, mask in enumerate(valid_masks_by_imgs)])  # N
        
        if self.training and mask2batch.size(0) == 0:
            _logger.warning("[Training] Number of valid masks in this local-batch is 0")
            # use sum all trainable parameters multiplied by 0
            loss = 0
            for module in self.saved_modules().values():
                for param in module.parameters():
                    if param.requires_grad:
                        loss += param.sum() * 0
            return loss
        
        encode_dict.update(self.visual_encoder(images, masks=valid_masks_by_imgs))  # List(B) of Q,D    
        
        corrs = self._get_corrs(encode_dict, valid_masks_by_imgs, mask2batch, t_embs)
        
        if self.training:
            encode_dict["mask2batch"] = mask2batch
            return self._get_loss(corrs, t_embs, valid_labels_by_imgs, encode_dict)
        
        # inference
        pred_logits = self._get_pred_logits_from_corr(corrs, mask2batch, valid_masks_by_imgs, num_synonyms)  # List(B) of Q,K

        if self.ensemble_on:
            assert valid_logits_by_imgs is not None
            mask_cls_results = self._ensemble_logits(pred_logits, valid_logits_by_imgs)
        else:
            mask_cls_results = pred_logits  # open-vocabulary logits from corr
        
        if valid_logits_by_imgs is not None:
            mask_cls_results = self._add_void_logits(mask_cls_results, valid_logits_by_imgs)
            
        mask_pred_results = valid_masks_by_imgs  # unpadded masks from segmentor
        del encode_dict
        
        processed_results = []
        for mask_cls_result, mask_pred_result, input_per_image in zip(
            mask_cls_results, mask_pred_results, batched_inputs
        ):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            processed_results.append({})
            processed_results[-1]["category_overlapping_mask"] = self.category_overlapping_mask
            
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(downsample_masks)(
                    mask_pred_result, (height, width), 'bilinear'
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)
            if self.mask_acc_on:
                processed_results[-1]["mask_cls"] = self._mask_cls_inferece(mask_cls_result)
            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self._semantic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(downsample_masks)(r, (height, width), 'bilinear')
                processed_results[-1]["sem_seg"] = r
            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self._panoptic_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r
            # instance segmentation inference
            if self.instance_on:
                instance_r = retry_if_cuda_oom(self._instance_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["instances"] = instance_r
        return processed_results
    
    
    def _mask_cls_inferece(self, mask_cls):
        K = len(self.test_num_synonyms)
        if mask_cls.size(1) == K:
            return mask_cls.argmax(-1)
        elif mask_cls.size(1) == K + 1:
            return mask_cls[:, :-1].argmax(-1)
        return None
        
    def _semantic_inference(self, mask_cls, mask_pred):
        K = len(self.test_num_synonyms)
        if mask_cls.size(1) == K:
            mask_cls = F.softmax(mask_cls, dim=-1)
        elif mask_cls.size(1) == K + 1:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def _panoptic_inference(self, mask_cls, mask_pred):
        K = len(self.test_num_synonyms)
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        num_classes = len(self.test_metadata.stuff_classes)
        keep = labels.ne(num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        if cur_mask_cls.size(1) == K + 1:
            cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.test_metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def _instance_inference(self, mask_cls, mask_pred):
        K = len(self.test_num_synonyms)
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        if mask_cls.size(1) == K:
            scores = F.softmax(mask_cls, dim=-1)
        elif mask_cls.size(1) == K + 1:
            scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # if this is panoptic segmentation
        if self.panoptic_on:
            num_classes = len(self.test_metadata.stuff_classes)
        else:
            num_classes = len(self.test_metadata.thing_classes)
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(scores.size(0), 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        test_topk_per_image = min(self.test_topk_per_image, scores.numel())
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        # topk_indices = topk_indices // num_classes
        topk_indices = torch.div(topk_indices, num_classes, rounding_mode='trunc')
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.test_metadata.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (after sigmoid)
        result.pred_masks = (mask_pred > 0.5).float()
        # Uncomment the following to get boxes from masks (this is slow)
        if self.instance_box_on:
            result.pred_boxes = BitMasks(mask_pred > 0.5).get_bounding_boxes()
        elif not self.run_demo:
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # calculate average mask prob
        mask_scores_per_image = (mask_pred.flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result