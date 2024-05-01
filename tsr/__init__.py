import os
import datetime
from modules import shared, util,devices
from modules.paths_internal import default_output_dir
from .system import TSR
from .utils import to_gradio_3d_orientation
import torch


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
        model.to(shared.device)
    finally:
        # restore disable_safe_unpickle to original value
        shared.cmd_opts.disable_safe_unpickle = disable_safe_unpickle



def generate(image, mc_resolution, formats=["obj", "glb"]):
    global model
    load_model()
    scene_codes = model(image, device=shared.device)
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
    reset_and_gc()
    return rv


def on_ui_settings():
    section = ('tsr', 'TripoSR')
    shared.opts.add_option(
        'tsr_output_dir',
        shared.OptionInfo(
            util.truncate_path(os.path.join(default_output_dir, 'triposr')),
            'Directory for TSR output models',
            section=section,
        )
    )
    shared.opts.add_option(
        'tsr_show_tips',
        shared.OptionInfo(
            True,
            'Show tips for TripoSR',
            section=section,
        ).needs_reload_ui()
    )

def reset_and_gc():
    import gc;gc.collect()
    devices.torch_gc()

    try:
        import os
        import psutil
        mem = psutil.Process(os.getpid()).memory_info()
        print(f'[Mem] rss: {mem.rss / 2 ** 30:.3f} GB, vms: {mem.vms / 2 ** 30:.3f} GB')
        from modules.shared import mem_mon as vram_mon
        from modules.memmon import MemUsageMonitor
        vram_mon: MemUsageMonitor
        free, total = vram_mon.cuda_mem_get_info()
        print(f'[VRAM] free: {free / 2 ** 30:.3f} GB, total: {total / 2 ** 30:.3f} GB')
    except:
        pass
