import os
import datetime
import torch
from modules import shared, util, devices
from modules.paths_internal import default_output_dir
from .system import TSR
from .utils import to_gradio_3d_orientation

device = devices.get_optimal_device_name()  # not sure if this works with other than CUDA or CPU
model = None


def load_model():
    global model
    if model is not None:
        return
    # --disable-safe-unpickle when loading tsr model
    disable_safe_unpickle = shared.cmd_opts.disable_safe_unpickle
    shared.cmd_opts.disable_safe_unpickle = True
    try:
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        # adjust the chunk size to balance between speed and memory usage
        model.renderer.set_chunk_size(8192)
        model.to(device)
    finally:
        # restore disable_safe_unpickle to original value
        shared.cmd_opts.disable_safe_unpickle = disable_safe_unpickle


def generate(image, mc_resolution, formats=["obj", "glb"]):
    load_model()
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    rv = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    for format in formats:
        # 生成文件名
        filename = f"file_{timestamp}.{format}"
        mesh_path = os.path.join(shared.opts.tsr_output_dir, filename)
        os.makedirs(shared.opts.tsr_output_dir, exist_ok=True)
        mesh.export(mesh_path)
        rv.append(mesh_path)
        print(f'model export in {mesh_path}')
    return rv


def on_ui_settings():
    shared.opts.add_option(
        'tsr_output_dir',
        shared.OptionInfo(
            util.truncate_path(os.path.join(default_output_dir, 'triposr')),
            'Directory for TSR output models',
            section=('tsr', 'TripoSR'),
        )
    )
