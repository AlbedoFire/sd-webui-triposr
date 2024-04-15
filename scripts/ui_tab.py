import gradio as gr
from modules import script_callbacks
from tsr.utils import preprocess
from tsr import generate, on_ui_settings


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def on_ui_tabs():
    with gr.Blocks(title="TripoSR") as interface:
        gr.Markdown(
            """
        # TripoSR Demo
        [TripoSR](https://github.com/VAST-AI-Research/TripoSR) is a state-of-the-art open-source model for **fast** feedforward 3D reconstruction from a single image, collaboratively developed by [Tripo AI](https://www.tripo3d.ai/) and [Stability AI](https://stability.ai/).

        **Tips:**
        1. If you find the result is unsatisfied, please try to change the foreground ratio. It might improve the results.
        2. It's better to disable "Remove Background" for the provided examples (except fot the last one) since they have been already preprocessed.
        3. Otherwise, please disable "Remove Background" option only if your input image is RGBA with transparent background, image contents are centered and occupy more than 70% of image width or height.
        """
        )
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(
                        label="Input Image",
                        image_mode="RGBA",
                        sources="upload",
                        type="pil",
                        elem_id="content_image",
                    )
                    processed_image = gr.Image(label="Processed Image", interactive=False)
                with gr.Row():
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
                with gr.Row():
                    submit = gr.Button("Generate", elem_id="generate", variant="primary")
            with gr.Column():
                with gr.Tab("OBJ"):
                    output_model_obj = gr.Model3D(
                        label="Output Model (OBJ Format)",
                        interactive=False,
                    )
                    gr.Markdown("Note: The model shown here is flipped. Download to get correct results.")
                with gr.Tab("GLB"):
                    output_model_glb = gr.Model3D(
                        label="Output Model (GLB Format)",
                        interactive=False,
                    )
                    gr.Markdown("Note: The obj model shown here has a darker appearance. Download to get correct results.")
        submit.click(fn=check_input_image, inputs=[input_image]).success(
            fn=preprocess,
            inputs=[input_image, do_remove_background, foreground_ratio],
            outputs=[processed_image],
        ).success(
            fn=generate,
            inputs=[processed_image, mc_resolution],
            outputs=[output_model_obj, output_model_glb],
        )
        return [(interface, "Extension TripoSR", "extension_TripoSR")]


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
