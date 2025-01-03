import gradio as gr
import os
import time
import glob
import cv2
import argparse
import numpy as np
from PIL import Image

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.projects.deeplab import add_deeplab_config
from maskclippp import add_maskformer2_config, add_maskclippp_config
from predictor import VisualizationDemo



def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskclippp_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.RUN_DEMO = True
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="maskclippp demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco-stuff/eva-clip-vit-l-14-336/maft-l/maskclippp_coco-stuff_eva-clip-vit-l-14-336_wtext_maft-l_ens.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def create_interface(meta_demo: VisualizationDemo) -> gr.Interface:
    predefined_classes_options = [
        "coco2017", "ade20k", "lvis1203", "cocostuff", "ade847", "ctx459", "ctx59", "voc20"
    ]
    with gr.Blocks() as demo:
        gr.Markdown("""### MaskCLIP++ Segmentation Demo
1. Choose the Predefined Classes and input the User Classes (if any).
2. Click on Submit Classes.
3. Upload an image.
4. Click on Submit Image to get the segmented image.
        """)
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input Image", image_mode="RGB")
                predefined_classes = gr.CheckboxGroup(choices=predefined_classes_options, label="Predefined Classes", value=["coco2017", "ade20k", "lvis1203"])
                user_classes = gr.Textbox(label="User Classes (e.g. 'tree,trees|sky,clouds')")
            with gr.Column():        
                set_classes_button = gr.Button("Submit Classes")
                status_text = gr.Textbox(label="Status", value="Classes not updated", interactive=False)
                output_image = gr.Image(type="pil", label="Output Image", image_mode="RGB")
                process_button = gr.Button("Submit Image")
    
        def update_classes(predefined_classes: list, user_classes: str):
            user_classes = user_classes.strip()
            if len(user_classes) > 0:
                user_classes_list = user_classes.split("|")
            else:
                user_classes_list = []
            logger.info("Predefined Classes: %s\nUser Classes: %s", predefined_classes, user_classes_list)
            return meta_demo.set_classes(predefined_classes, user_classes_list)
            
        def process_image(image):
            start_time = time.time()
            bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            predictions, visualized_output = meta_demo.run_on_image(bgr_image)
            logger.info(
                "{} in {:.2f}s".format(
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            return Image.fromarray(visualized_output.get_image())
        
        set_classes_button.click(fn=update_classes,
                                 inputs=[predefined_classes, user_classes],
                                 outputs=[status_text])
        process_button.click(fn=process_image,
                             inputs=[image_input],
                             outputs=[output_image])
    return demo

if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    meta_demo = VisualizationDemo(cfg, confidence_threshold=args.confidence_threshold)
    gradio_demo = create_interface(meta_demo)
    gradio_demo.queue()
    gradio_demo.launch()
    