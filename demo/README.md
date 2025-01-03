# Use the demo of MaskCLIP++

<!-- ![](../assets/image1.gif)
![](../assets/image2.gif)
![](../assets/avengers.gif) -->

<p align="center">
  <img src="../assets/image1.gif" alt="image1" width="600"/>
</p>
<p align="center">
  <img src="../assets/image2.gif" alt="image2" width="300"/>
</p>
<p align="center">
  <img src="../assets/avengers.gif" alt="avengers" width="600"/>
</p>


We provide two forms of demo. A gradio based `demo/app.py` for processing individual images and adjusting vocabularies. One is `demo/demo.py`, which is used on the command line and is suitable for batch processing multiple pictures or videos. The instructions are as follows:


## Gradio Demo

1. Select the model and launch the application using the following command
2. Choose the `Predefined Classes` and input the `User Classes` (if any).
3. Click on `Submit Classes`.
4. Upload an image.
5. Click on `Submit Image` to get the segmented image.

### Open-Vocabulary Semantic Segmentation

```shell
config="configs/coco-stuff/eva-clip-vit-l-14-336/maft-l/maskclippp_coco-stuff_eva-clip-vit-l-14-336_wtext_maft-l_ens.yaml"
ckpt="output/ckpts/maskclippp/maskclippp_coco-stuff_eva-clip-vit-l-14-336_wtext.pth"
python demo/app.py \
    --config-file $config \
    --opts \
    MODEL.WEIGHTS $ckpt \
    MODEL.MASK_FORMER.TEST.PANOPTIC_ON False \
    MODEL.MASK_FORMER.TEST.INSTANCE_ON False \
    MODEL.MASK_FORMER.TEST.SEMANTIC_ON True
```

### Open-Vocabulary Panoptic Segmentation

**Note: Please use the mask generator trained on panoptic segmentation data**

```shell
config="configs/coco-pan/eva-clip-vit-l-14-336/fcclip-l/maskclippp_coco-pan_eva-clip-vit-l-14-336_fcclip-l.yaml"
ckpt="output/ckpts/maskclippp/maskclippp_coco-pan_eva-clip-vit-l-14-336.pth"
python demo/app.py \
    --config-file $config \
    --opts \
    MODEL.WEIGHTS $ckpt \
    MODEL.MASK_FORMER.TEST.PANOPTIC_ON True \
    MODEL.MASK_FORMER.TEST.INSTANCE_ON False \
    MODEL.MASK_FORMER.TEST.SEMANTIC_ON False
```

### Open-Vocabulary Instance Segmentation

**Note: Please use the mask generator trained on panoptic/instance segmentation data**

```shell
config="configs/coco-pan/eva-clip-vit-l-14-336/fcclip-l/maskclippp_coco-pan_eva-clip-vit-l-14-336_fcclip-l.yaml"
ckpt="output/ckpts/maskclippp/maskclippp_coco-pan_eva-clip-vit-l-14-336.pth"
python demo/app.py \
    --config-file $config \
    --confidence
    --opts \
    MODEL.WEIGHTS $ckpt \
    MODEL.MASK_FORMER.TEST.PANOPTIC_ON True \
    MODEL.MASK_FORMER.TEST.INSTANCE_ON True \
    MODEL.MASK_FORMER.TEST.SEMANTIC_ON False
```

## Command Demo

Similar to the `demo/app.py` startup command, except that the input, output, and vocabularies need to be specified through the command line. For details please see:
```
python demo/demo.py -h
```