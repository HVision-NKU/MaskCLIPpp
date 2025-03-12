# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import numpy as np
import os
from collections import OrderedDict
from typing import Optional, Dict, List, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image
from tabulate import tabulate
from panopticapi.utils import rgb2id

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation import DatasetEvaluator


def load_image_into_numpy_array(
    filename: str,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    with PathManager.open(filename, "rb") as f:
        array = np.asarray(Image.open(f), dtype=dtype)
    return array


class MaskAccEvaluator(DatasetEvaluator):
    """
    Evaluate mask accuracy by matching predicted masks with ground truth masks using Hungarian algorithm based on IoU,
    then compute the accuracy of matched pairs based on class predictions.
    Should be used with GTSegmentor that load from ground truth masks.
    """

    def __init__(
        self,
        dataset_name: str,
        distributed: bool = True,
        output_dir: Optional[str] = None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
            output_dir (str): an output directory to dump results.
        """
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }
        
        self.sem_seg_loading_fn = sem_seg_loading_fn
        
        # Metadata
        meta = MetadataCatalog.get(dataset_name)
        self._contiguous_id_to_dataset_id = None
        # try:
        #     c2d = meta.stuff_dataset_id_to_contiguous_id
        #     self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        # except AttributeError:
        #     pass
        self._class_names = meta.stuff_classes
        self._num_classes = len(meta.stuff_classes)
        self._ignore_label = meta.ignore_label

        self.reset()

    def reset(self) -> None:
        # self.total_correct = np.zeros(self._num_classes, dtype=int)
        # self.total_matched = np.zeros(self._num_classes, dtype=int)
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._category_overlapping_mask = None
        
        
    def _load_gt_classes(self, one_input):
        gt_filename = self.input_file_to_gt_file[one_input["file_name"]]
        gt = self.sem_seg_loading_fn(gt_filename, dtype=int)
        gt_classes = np.unique(gt)
        gt_classes = gt_classes[gt_classes != self._ignore_label]
        return gt_classes
        
        
    def process(self, inputs: List[Dict], outputs: List[Dict]) -> None:
        """
        Process inputs and outputs to compute matching statistics.
        """
        if self._category_overlapping_mask is None:
            for output in outputs:
                if "category_overlapping_mask" in output:
                    self._category_overlapping_mask = output["category_overlapping_mask"].to(self._cpu_device).numpy()
                    assert len(self._category_overlapping_mask) == self._num_classes
                    break
        for input, output in zip(inputs, outputs):
            # Extract predictions
            pred_classes = output["mask_cls"].to(self._cpu_device).numpy()  # (N,)
            gt_classes = self._load_gt_classes(input)
            
            if len(pred_classes) == 0 or len(gt_classes) == 0:
                continue
            
            assert len(pred_classes) == len(gt_classes), f"Number of predictions {len(pred_classes)} != number of GT classes {len(gt_classes)}"
            
            for pred_cls, gt_cls in zip(pred_classes, gt_classes):
                if self._contiguous_id_to_dataset_id is not None:
                    pred_cls = self._contiguous_id_to_dataset_id[pred_cls]
                self._conf_matrix[pred_cls, gt_cls] += 1
            

    def _log_table(self, classes, results):
        results_per_category = []
        for name in classes:
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            one_name = name.split(',')[0]
            results_per_category.append((str(one_name), 
                                         float(results["mask_acc"][f"MaskAcc-{name}"])))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "MaskAcc"] * (N_COLS // 2),
            numalign="left",
        )
        return table
    
    def evaluate(self) -> OrderedDict:
        """
        Synchronize and compute accuracy metrics.
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        acc = np.full(self._num_classes, np.nan, dtype=float)
        tp = self._conf_matrix.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        
        if self._category_overlapping_mask is not None:
            num_seen = np.sum(self._category_overlapping_mask)
            if num_seen == 0:
                self._logger.info("All categories are unseen.")
                macc_seen = macc_unseen = None
            elif num_seen == self._num_classes:
                self._logger.info("All categories are seen.")
                macc_seen = macc_unseen = None
            else:
                macc_valid_seen = np.logical_and(acc_valid, self._category_overlapping_mask)
                macc_valid_unseen = np.logical_and(acc_valid, ~self._category_overlapping_mask)
                macc_seen = np.sum(acc[macc_valid_seen]) / np.sum(macc_valid_seen)
                macc_unseen = np.sum(acc[macc_valid_unseen]) / np.sum(macc_valid_unseen)
        else:
            macc_seen = macc_unseen = None
        
        res = {}
        res["mMaskAcc"] = 100 * macc
        if macc_seen is not None:
            res["mMaskAcc(S)"] = 100 * macc_seen
        if macc_unseen is not None:
            res["mMaskAcc(U)"] = 100 * macc_unseen
        for i, cls_name in enumerate(self._class_names):
            res[f"MaskAcc-{cls_name}"] = 100 * acc[i]
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "mask_acc_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"mask_acc": res})
        
        if macc_seen is not None and macc_unseen is not None:
            seen_class_names = [name for i, name in enumerate(self._class_names) if self._category_overlapping_mask[i]]
            unseen_class_names = [name for i, name in enumerate(self._class_names) if not self._category_overlapping_mask[i]]
            self._logger.info("Mask accuracy on seen classes:\n%s", self._log_table(seen_class_names, results))
            self._logger.info("Mask accuracy on unseen classes:\n%s", self._log_table(unseen_class_names, results))
            self._logger.info("mMaskAcc(S) = %.3f, mMaskAcc(U) = %.3f, mMaskAcc = %.3f", macc_seen * 100, macc_unseen * 100, macc * 100)
        else:
            self._logger.info("Mask accuracy on all classes:\n%s", self._log_table(self._class_names, results))
            self._logger.info("mMaskAcc = %.3f", macc * 100)
        # self._logger.info(results)
        return results


class PanMaskAccEvaluator(MaskAccEvaluator):
    def __init__(
        self,
        dataset_name: str,
        distributed: bool = True,
        output_dir: Optional[str] = None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
    ):
        super().__init__(dataset_name, distributed, output_dir, sem_seg_loading_fn=sem_seg_loading_fn)
        self.input_file_to_segments_info = {
            dataset_record["file_name"]: dataset_record["segments_info"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }
        
    
    def _load_gt_classes(self, one_input):
        segments_info = self.input_file_to_segments_info[one_input["file_name"]]
        gt_classes = [seg["category_id"] for seg in segments_info]
        return np.array(gt_classes)
