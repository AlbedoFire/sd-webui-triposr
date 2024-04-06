# Triposr

Extension for [webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

a extensions bring [TripoSr Demo](https://huggingface.co/spaces/stabilityai/TripoSR) to webui

# Installation

1.Install from webui's Extensions tab.  
2.You need to add --disable-safe-unpickle commandline argument to webui-user.bat

## Troubleshooting
> AttributeError: module 'torchmcubes_module' has no attribute 'mcubes_cuda'

or

> torchmcubes was not compiled with CUDA support, use CPU version instead.

This is because `torchmcubes` is compiled without CUDA support. Please make sure that 

- The locally-installed CUDA major version matches the PyTorch-shipped CUDA major version. For example if you have CUDA 11.x installed, make sure to install PyTorch compiled with CUDA 11.x.
- `setuptools>=49.6.0`. If not, upgrade by `pip install --upgrade setuptools`.

Then re-install `torchmcubes` by:

```sh
pip uninstall torchmcubes
pip install git+https://github.com/tatsy/torchmcubes.git
```

# How to Use
1.You can use the tab just like [TripoSr Demo](https://huggingface.co/spaces/stabilityai/TripoSR)  
2.You can use in txt2img tab it will generate model in outputs folder

# Credits

https://github.com/VAST-AI-Research/TripoSR.git
