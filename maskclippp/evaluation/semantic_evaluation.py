import logging
import os
import json
import itertools
from collections import OrderedDict
from typing import Optional, Union
from detectron2.evaluation.sem_seg_evaluation import load_image_into_numpy_array
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tabulate import tabulate

from detectron2.data import MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import SemSegEvaluator as _SemSegEvaluator


_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False


def load_image_into_numpy_array(
    filename: str,
    copy: bool = False,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    with PathManager.open(filename, "rb") as f:
        array = np.array(Image.open(f), copy=copy, dtype=dtype)
    return array


class SemSegEvaluator(_SemSegEvaluator):
    def __init__(self, dataset_name, distributed=True, output_dir=None, *, 
                 sem_seg_loading_fn=load_image_into_numpy_array, num_classes=None, ignore_label=None):
        super().__init__(dataset_name, distributed, output_dir, sem_seg_loading_fn=sem_seg_loading_fn, num_classes=num_classes, ignore_label=ignore_label)
        self._category_overlapping_mask = None
        
    def process(self, inputs, outputs):
        super().process(inputs, outputs)
        if self._category_overlapping_mask is None:
            for output in outputs:
                if "category_overlapping_mask" in output:
                    self._category_overlapping_mask = output["category_overlapping_mask"].to(self._cpu_device).numpy()
                    assert len(self._category_overlapping_mask) == self._num_classes
                    break
    
    def _log_table(self, classes, results):
        results_per_category = []
        for name in classes:
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            one_name = name.split(',')[0]
            results_per_category.append((str(one_name), 
                                         float(results["sem_seg"][f"IoU-{name}"]),
                                         float(results["sem_seg"][f"ACC-{name}"])))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "IoU", "ACC"] * (N_COLS // 2),
            numalign="left",
        )
        return table
    
    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            b_conf_matrix_list = all_gather(self._b_conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

            self._b_conf_matrix = np.zeros_like(self._b_conf_matrix)
            for b_conf_matrix in b_conf_matrix_list:
                self._b_conf_matrix += b_conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)
        tp = self._conf_matrix.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)
        
        if self._category_overlapping_mask is not None:
            num_seen = np.sum(self._category_overlapping_mask)
            if num_seen == 0:
                self._logger.info("All categories are unseen.")
                miou_seen = miou_unseen = None
            elif num_seen == self._num_classes:
                self._logger.info("All categories are seen.")
                miou_seen = miou_unseen = None
            else:
                iou_valid_seen = np.logical_and(iou_valid, self._category_overlapping_mask)
                iou_valid_unseen = np.logical_and(iou_valid, ~self._category_overlapping_mask)
                miou_seen = np.sum(iou[iou_valid_seen]) / np.sum(iou_valid_seen)
                miou_unseen = np.sum(iou[iou_valid_unseen]) / np.sum(iou_valid_unseen)
        else:
            miou_seen = miou_unseen = None
            
        if self._compute_boundary_iou:
            b_iou = np.full(self._num_classes, np.nan, dtype=float)
            b_tp = self._b_conf_matrix.diagonal()[:-1].astype(float)
            b_pos_gt = np.sum(self._b_conf_matrix[:-1, :-1], axis=0).astype(float)
            b_pos_pred = np.sum(self._b_conf_matrix[:-1, :-1], axis=1).astype(float)
            b_union = b_pos_gt + b_pos_pred - b_tp
            b_iou_valid = b_union > 0
            b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]

        res = {}
        res["mIoU"] = 100 * miou
        if miou_seen is not None:
            res["mIoU(S)"] = 100 * miou_seen
        if miou_unseen is not None:
            res["mIoU(U)"] = 100 * miou_unseen
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
            if self._compute_boundary_iou:
                res[f"BoundaryIoU-{name}"] = 100 * b_iou[i]
                res[f"min(IoU, B-Iou)-{name}"] = 100 * min(iou[i], b_iou[i])
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}"] = 100 * acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        # self._logger.info(results)
        if miou_seen is not None and miou_unseen is not None:
            seen_class_names = [name for i, name in enumerate(self._class_names) if self._category_overlapping_mask[i]]
            unseen_class_names = [name for i, name in enumerate(self._class_names) if not self._category_overlapping_mask[i]]
            self._logger.info("Seen Category Results:\n%s", self._log_table(seen_class_names, results))
            self._logger.info("Unseen Category Results:\n%s", self._log_table(unseen_class_names, results))
            self._logger.info("mIoU(S) = %.3f, mIoU(U) = %.3f", miou_seen * 100, miou_unseen * 100)
        else:
            self._logger.info("All Category Results:\n%s", self._log_table(self._class_names, results))
            self._logger.info("mIoU = %.3f", miou * 100)
        return results
    
    