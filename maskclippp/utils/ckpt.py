from typing import Any, IO, cast, Dict, Sequence
import logging
import os
import numpy as np
import pickle
import gdown
import requests
from tqdm import tqdm
from urllib.parse import parse_qs, urlparse
import torch
from torch import nn
from fvcore.common.checkpoint import Checkpointer
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.utils.file_io import PathManager
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts


MASK_GENERATOR_URLS = {
    "maskformer2_swin_tiny_bs16_50ep_final_9fd0ae": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_tiny_bs16_50ep/model_final_9fd0ae.pkl",
    "maskformer2_swin_large_IN21k_384_bs16_100ep_final_f07440": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl",
    "fcclip_coco-pan_clip-convnext-base": "https://drive.google.com/uc?id=1fSFPPTwxF-ekMxAmIo01ssdbC79wwwml",
    "fcclip_coco-pan_clip-convnext-large": "https://drive.google.com/uc?id=1-91PIns86vyNaL3CzMmDD39zKGnPMtvj",
    "maftp_b": "https://drive.google.com/uc?id=1BeEeKOnWWIWIH-QWK_zLhAPUzCOnHuFG",
    "maftp_l": "https://drive.google.com/uc?id=1EQo5guVuKkSSZj4bv0FQN_4X9h_Rwfe5",
    "maftp_l_pano": "https://drive.google.com/uc?id=1znk_uco8fwvbA0kndy4kGyVp22KbQr6g"
}


def download_with_progress(url, save_path):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    tmp_path = save_path + ".tmp"
    file_name = os.path.basename(save_path)
    
    with requests.get(url, headers=headers, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        progress = tqdm(
            total=total_size, 
            unit='B',
            unit_scale=True,
            desc=f"Downloading {file_name}",
        )

        with open(tmp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                progress.update(len(chunk))
        progress.close()
        # rename the temp file to the final file
        os.rename(tmp_path, save_path)




def download_mask_generator(path: str, logger: logging.Logger):
    if os.path.exists(path):
        logger.info("Mask generator already exists at %s", path)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    basename = os.path.splitext(os.path.basename(path))[0]
    if basename not in MASK_GENERATOR_URLS:
        raise ValueError(f"Cannot get url for {basename}")
    logger.info("Begin downloading mask generator from %s", MASK_GENERATOR_URLS[basename])
    url = MASK_GENERATOR_URLS[basename]
    if "drive.google.com" in url:
        gdown.download(url, path, quiet=False)
    else:
        download_with_progress(url, path)
    logger.info("Downloaded mask generator to %s", path)
    
    
    

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