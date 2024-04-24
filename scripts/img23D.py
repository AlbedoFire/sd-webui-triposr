import gradio as gr
import modules.scripts as scripts
from modules.processing import process_images
from tsr.utils import preprocess
from tsr import generate


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


class TripoSR(scripts.Script):

    def title(self):
        return "TripoSR image to 3D"

    # Determines when the script should be shown in the dropdown menu via the
    # returned value. As an example:
    # is_img2img is True if the current tab is img2img, and False if it is txt2img.
    # Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):
        return True

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
        proc = process_images(p)
        images_list = proc.images
        for i in images_list:
            proces_img = preprocess(i, do_remove_background, foreground_ratio)
            generate(proces_img, mc_resolution)

        # TODO: add image edit process via Processed object proc
        return proc
