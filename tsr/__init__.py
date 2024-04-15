import os
import datetime
import torch
from modules import shared
from .system import TSR
from .utils import to_gradio_3d_orientation


save_path = 'outputs/triposr'  # todo: need change

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
    for format in formats:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # 生成文件名
        filename = f"file_{timestamp}.{format}"
        mesh_path = os.path.join(save_path, filename)
        mesh.export(mesh_path)
        rv.append(mesh_path)
        print(f'model export in {mesh_path}')
    return rv
