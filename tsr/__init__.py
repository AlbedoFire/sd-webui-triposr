import os
import datetime
import torch
from modules import shared, util
from modules.paths_internal import default_output_dir
from .system import TSR
from .utils import to_gradio_3d_orientation


device = shared.cmd_opts.device_id if torch.cuda.is_available() else "cpu"

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(8192)
model.to(device)


def generate(image, mc_resolution, formats=["obj", "glb"]):
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
            section=('tsr', 'TSR'),
        )
    )
