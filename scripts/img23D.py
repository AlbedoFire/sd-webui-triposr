import datetime
import sys

import modules.scripts as scripts
import gradio as gr
import os

from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state

import logging
import os
import tempfile
import time

import gradio as gr
import numpy as np
import rembg
import torch
from PIL import Image
from functools import partial

# 获取当前文件的路径
current_path = os.path.dirname(os.path.dirname(__file__))
print('Current path: {}'.format(current_path))
sys.path.append(current_path)

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

import argparse



if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(8192)
model.to(device)

rembg_session = rembg.new_session()

save_path = 'outputs/triposr'
def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background, foreground_ratio):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image


def generate(image, mc_resolution, formats=["obj", "glb"]):
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    rv = []
    for format in formats:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # 生成文件名
        filename = f"file_{timestamp}.txt"
        mesh_path = os.path.join(save_path,filename)
        mesh.export(mesh_path)
        print('model export in {}'.format(mesh_path))







class TripoSR(scripts.Script):

    def title(self):
        return "图片转化为3D模型"

    # Determines when the script should be shown in the dropdown menu via the
    # returned value. As an example:
    # is_img2img is True if the current tab is img2img, and False if it is txt2img.
    # Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # How the script's is displayed in the UI. See https://gradio.app/docs/#components
    # for the different UI components you can use and how to create them.
    # Most UI components can return a value, such as a boolean for a checkbox.
    # The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        with gr.Group():
            do_remove_background = gr.Checkbox(
                label="Remove Background", value=True
            )
            foreground_ratio = gr.Slider(
                label="Foreground Ratio",
                minimum=0.5,
                maximum=1.0,
                value=0.85,
                step=0.05,
            )
            mc_resolution = gr.Slider(
                label="Marching Cubes Resolution",
                minimum=32,
                maximum=320,
                value=256,
                step=32
            )
        return [do_remove_background, foreground_ratio, mc_resolution]

    # This is where the additional processing is implemented. The parameters include
    # self, the model object "p" (a StableDiffusionProcessing class, see
    # processing.py), and the parameters returned by the ui method.
    # Custom functions can be defined here, and additional libraries can be imported
    # to be used in processing. The return value should be a Processed object, which is
    # what is returned by the process_images method.

    def run(self, p, do_remove_background, foreground_ratio, mc_resolution):
        # function which takes an image from the Processed object,
        # and the angle and two booleans indicating horizontal and
        # vertical flips from the UI, then returns the
        # image rotated and flipped accordingly
        def rotate_and_flip(im, angle, hflip, vflip):
            proc = process_images(p)
            images_list = proc.images
            for i in images_list:
                proces_img = preprocess(i, do_remove_background, foreground_ratio)
                generate(proces_img, mc_resolution)

            # TODO: add image edit process via Processed object proc
            return proc
