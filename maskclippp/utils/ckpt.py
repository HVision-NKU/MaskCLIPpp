from typing import Any, IO, cast, Dict, Sequence
import logging
import os
import numpy as np
import pickle
from urllib.parse import parse_qs, urlparse
import torch
from torch import nn
from fvcore.common.checkpoint import Checkpointer
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.utils.file_io import PathManager
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts


def load_state_dict_with_beg_key(module: nn.Module, 
                                 state_dict: Dict[str, torch.Tensor], 
                                 beg_key: str, 
                                 module_name: str,
                                 file_path: str,
                                 logger: logging.Logger) -> None:
    selected_dict = {}
    for key, value in state_dict.items():
        if key.startswith(beg_key):
            selected_dict[key[len(beg_key):]] = value
    missing_keys, unexpected_keys = module.load_state_dict(selected_dict, strict=False)
    if len(missing_keys) > 0:
        logger.warning("Missing keys when loading weights of %s from %s in file %s:\n%s",
                        module_name, beg_key, file_path, missing_keys)
    if len(unexpected_keys) > 0:
        logger.warning("Unexpected keys when loading weights of %s from %s in file %s:\n%s",
                        module_name, beg_key, file_path, unexpected_keys)
    logger.info("Loaded weights of %s from %s in file %s", module_name, beg_key, file_path)


def convert_ndarray_to_tensor(state_dict: Dict[str, Any]) -> None:
        """
        In-place convert all numpy arrays in the state_dict to torch tensor.
        Args:
            state_dict (dict): a state-dict to be loaded to the model.
                Will be modified.
        """
        # model could be an OrderedDict with _metadata attribute
        # (as returned by Pytorch's state_dict()). We should preserve these
        # properties.
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
                raise ValueError(
                    "Unsupported type found in checkpoint! {}: {}".format(k, type(v))
                )
            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.from_numpy(v)

class MaskCorrCheckpointer(DetectionCheckpointer):
    def save(self, name: str, **kwargs: Any) -> None:
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        if hasattr(self.model, "saved_modules"):
            model_state_dict = {}
            for k, v in self.model.saved_modules().items():
                for k_, v_ in v.state_dict().items():
                    model_state_dict["{}.{}".format(k, k_)] = v_
            data["model"] = model_state_dict
        else:
            data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, cast(IO[bytes], f))
        self.tag_last_checkpoint(basename)