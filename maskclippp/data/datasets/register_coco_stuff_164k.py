"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/MendelXu/SAN/blob/main/san/data/datasets/register_coco_stuff_164k.py
"""

import os
import logging
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from functools import partial

from . import openseg_classes
from ...utils.misc import read_file_list

_logger = logging.getLogger(__name__)

COCO_CATEGORIES = openseg_classes.get_coco_stuff_categories_with_prompt_eng()


def load_partial_sem_seg(gt_root, image_root, gt_suffix=".png", image_suffix=".jpg", split=None):
    # for debug
    if split is None:
        return load_sem_seg(gt_root, image_root, gt_ext=gt_suffix[1:], image_ext=image_suffix[1:])

    file_list = read_file_list(split_path=split, default_dir=gt_root, default_suffix=gt_suffix)
    _logger.info(
        "Loaded {} images with semantic segmentation according to {}".format(len(file_list), gt_root if split is None else split)
    )
    dataset_dicts = []
    for name in file_list:
        record = {}
        record["file_name"] = os.path.join(image_root, name + image_suffix)
        record["sem_seg_file_name"] = os.path.join(gt_root, name + gt_suffix)
        dataset_dicts.append(record)
    return dataset_dicts

def _get_coco_stuff_meta(zsl=False):
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    # stuff_ids = [k["id"] for k in COCO_CATEGORIES]
    stuff_ids = list(range(len(COCO_CATEGORIES)))  # ids are already contigious because stuffthingmaps_detectron2 are used.
    unseen_ids = [19, 23, 28, 29, 36, 51, 76, 88, 
                  94, 112, 133, 136, 137, 157, 160]
    # seen_ids = np.setdiff1d(stuff_ids, unseen_ids).tolist()
    assert len(stuff_ids) == 171, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    
    if zsl:
        ret["unseen_ids"] = unseen_ids
    return ret


def register_all_coco_stuff_164k(root):
    root = os.path.join(root, "coco")
    meta = _get_coco_stuff_meta()

    for name, image_dirname, sem_seg_dirname in [
        ("train", "train2017", "stuffthingmaps_detectron2/train2017"),
        ("val", "val2017", "stuffthingmaps_detectron2/val2017"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        all_name = f"openvocab_coco_2017_{name}_stuff_sem_seg"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


def register_partial_coco_stuff(root, split_path):
    
    root = os.path.join(root, "coco")
    meta = _get_coco_stuff_meta()
    image_dir = os.path.join(root, "train2017")
    gt_dir = os.path.join(root, "stuffthingmaps_detectron2/train2017")
    dname = os.path.basename(split_path).split(".")[0]
    all_name = f"openvocab_coco_2017_{dname}_stuff_sem_seg"
    
    DatasetCatalog.register(
        all_name,
        lambda x=image_dir, y=gt_dir: load_partial_sem_seg(
            y, x, gt_suffix=".png", image_suffix=".jpg", split=split_path
        )
    )
    MetadataCatalog.get(all_name).set(
        image_root=image_dir,
        sem_seg_root=gt_dir,
        evaluator_type="sem_seg",
        ignore_label=255,
        **meta,
    )
        
    

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_stuff_164k(_root)

# partial_train_sets_dir = os.path.join(_root, "coco/sets/partial_train")
# partial_train_sets = os.listdir(partial_train_sets_dir)
# for split in partial_train_sets:
#     split_path = os.path.join(partial_train_sets_dir, split)
#     register_partial_coco_stuff(_root, split_path)