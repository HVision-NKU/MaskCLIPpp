## Installation

### Requirements
- Linux with Python = 3.10
- CUDA = 12.1, PyTorch = 2.3.1 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- Install Detectron2 from source: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- `pip install -r requirements.txt`
- `pip install git+https://github.com/cocodataset/panopticapi.git`


### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd maskclippp/segmentor/ops
sh make.sh
```

### Example conda environment setup
```bash
# conda create --name maskclippp python=3.10 -y
# conda activate maskclippp
# # CUDA 12.1
# pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# pip install git+https://github.com/cocodataset/panopticapi.git

# git clone git@github.com:HVision-NKU/MaskCLIPpp.git
# cd maskclippp
# pip install -r requirements.txt
# cd maskclippp/segmentor/ops
# sh make.sh
# cd ../../../
```
