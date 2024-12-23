import itertools
import json
import logging
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
from PIL import Image
from panopticapi.utils import rgb2id
from collections import OrderedDict
from detectron2.structures import Instances, Boxes, BitMasks
from detectron2.evaluation import DatasetEvaluator, COCOPanopticEvaluator
from .semantic_evaluation import SemSegEvaluator
from .instance_evaluation import InstanceSegEvaluator

from ..utils.misc import mask_iou



class MaskWithGTAssignEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        semantic_on,
        panoptic_on,
        instance_on,
        distributed=True,
        output_dir=None,
        ):
        
        assert semantic_on or panoptic_on or instance_on
        
        self._logger = logging.getLogger(__name__)
        self._cpu_device = torch.device("cpu")
        self._semantic_on = semantic_on
        self._panoptic_on = panoptic_on
        self._instance_on = instance_on
        if self._semantic_on:
            self._proxy_semseg = SemSegEvaluator(dataset_name, distributed=distributed, output_dir=output_dir)
        if self._panoptic_on:
            self._proxy_panseg = COCOPanopticEvaluator(dataset_name, output_dir=output_dir)
        if self._instance_on:
            self._proxy_insseg = InstanceSegEvaluator(dataset_name, output_dir=output_dir)
            self._thing_dataset_id_map = {id: i for i, id in enumerate(self._proxy_insseg._metadata.thing_dataset_id_to_contiguous_id.keys())}
        
    def reset(self):
        if self._semantic_on:
            self._proxy_semseg.reset()
        if self._panoptic_on:
            self._proxy_panseg.reset()
        if self._instance_on:
            self._proxy_insseg.reset()
            
    def _semseg_process(self, inputs, outputs):
        for ipt, output in zip(inputs, outputs):
            pred_masks = output["pred_masks"].to(self._cpu_device).numpy()
            gt_filename = self._proxy_semseg.input_file_to_gt_file[ipt["file_name"]]
            gt_ann = np.array(Image.open(gt_filename))
            gt_labels = np.unique(gt_ann)
            gt_labels = gt_labels[gt_labels != self._proxy_semseg._ignore_label]
            gt_masks = []
            for label in gt_labels:
                gt_masks.append(gt_ann == label)
            if len(gt_masks) > 0:
                gt_masks = np.stack(gt_masks, axis=0)
                iou_mat = mask_iou(pred_masks, gt_masks, scale=0.25)
                pred_indices, gt_indices = linear_sum_assignment(iou_mat, maximize=True)
                sem_seg = np.zeros((self._proxy_semseg._num_classes, *pred_masks.shape[1:]), dtype=np.float32)
                for pred_idx, gt_idx in zip(pred_indices, gt_indices):
                    sem_seg[gt_labels[gt_idx]] = pred_masks[pred_idx]
                output["sem_seg"] = torch.from_numpy(sem_seg)
            else:
                # This should not happen
                raise ValueError(f"No ground truth masks found in {gt_filename}")
        self._proxy_semseg.process(inputs, outputs)
        
    def _panseg_process(self, inputs, outputs):
        for ipt, output in zip(inputs, outputs):
            pred_masks = output["pred_masks"].to(self._cpu_device).numpy()
            gt_filename = ipt["pan_seg_file_name"]
            gt_ann = rgb2id(np.asarray(Image.open(gt_filename)))
            gt_info = ipt["segments_info"]
            gt_masks = []
            for seg in gt_info:
                gt_masks.append(gt_ann == seg["id"])
            if len(gt_masks) > 0:
                gt_masks = np.stack(gt_masks, axis=0)
                iou_mat = mask_iou(pred_masks, gt_masks, scale=0.25)
                pred_indices, gt_indices = linear_sum_assignment(iou_mat, maximize=True)
                pan_seg = np.zeros_like(gt_ann)
                seg_info = []
                idx = 0
                for pred_idx, gt_idx in zip(pred_indices, gt_indices):
                    pan_seg[pred_masks[pred_idx]] = idx
                    seg_info.append(dict(id=idx, 
                                         category_id=gt_info[gt_idx]["category_id"],
                                         isthing=gt_info[gt_idx]["isthing"],))
                    idx += 1
                remained_seg_info = []
                for i in range(idx):
                    if pan_seg[pan_seg == i].sum() > 0:
                        remained_seg_info.append(seg_info[i])
                pan_seg = torch.from_numpy(pan_seg)
                output["panoptic_seg"] = (pan_seg, remained_seg_info)
            else:
                pan_seg = torch.from_numpy(np.zeros_like(gt_ann))
                output["panoptic_seg"] = (pan_seg, [])
        self._proxy_panseg.process(inputs, outputs)
            
    def _insseg_process(self, inputs, outputs):
        # This is the version using panseg
        for ipt, output in zip(inputs, outputs):
            pred_masks = output["pred_masks"].to(self._cpu_device).numpy()
            gt_filename = ipt["pan_seg_file_name"]
            gt_ann = rgb2id(np.asarray(Image.open(gt_filename)))
            gt_info = ipt["segments_info"]
            gt_info = [x for x in gt_info if x["isthing"]]
            gt_masks = []
            for seg in gt_info:
                gt_masks.append(gt_ann == seg["id"])
            if len(gt_masks) > 0:
                gt_masks = np.stack(gt_masks, axis=0)
                iou_mat = mask_iou(pred_masks, gt_masks, scale=0.25)
                pred_indices, gt_indices = linear_sum_assignment(iou_mat, maximize=True)
                result = Instances(gt_ann.shape)
                result.pred_masks = BitMasks(torch.from_numpy(pred_masks[pred_indices]))
                result.pred_boxes = result.pred_masks.get_bounding_boxes()
                pred_classes = []
                for gid in gt_indices:
                    cid = gt_info[gid]["category_id"]
                    # cid = self._thing_dataset_id_map[cid]
                    pred_classes.append(cid)
                result.pred_classes = torch.tensor(pred_classes, dtype=torch.int64)
                result.scores = torch.tensor([1.0] * len(pred_classes), dtype=torch.float32)
            else:
                result = Instances(gt_ann.shape)
                result.pred_masks = BitMasks(torch.zeros((0, *gt_ann.shape)))
                result.pred_boxes = Boxes(torch.zeros((0, 4)))
                result.pred_classes = torch.zeros((0,), dtype=torch.int64)
                result.scores = torch.zeros((0,), dtype=torch.float32)
            output["instances"] = result
        self._proxy_insseg.process(inputs, outputs)
        
        
    def process(self, inputs, outputs):
        if self._semantic_on:
            self._semseg_process(inputs, outputs)
        if self._panoptic_on:
            self._panseg_process(inputs, outputs)
        if self._instance_on:
            self._insseg_process(inputs, outputs)

    def evaluate(self):
        rtn_dict = OrderedDict({})
        if self._semantic_on:
            sem_seg_rtn = self._proxy_semseg.evaluate()
            if sem_seg_rtn is not None:
                rtn_dict.update(sem_seg_rtn)
        if self._panoptic_on:
            pan_seg_rtn = self._proxy_panseg.evaluate()
            if pan_seg_rtn is not None:
                rtn_dict.update(pan_seg_rtn)
        if self._instance_on:
            ins_seg_rtn = self._proxy_insseg.evaluate()
            if ins_seg_rtn is not None:
                rtn_dict.update(ins_seg_rtn)
        return rtn_dict