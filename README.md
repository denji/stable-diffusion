# CompVis Stable-Diffusion

Custom installation to combine all of the best work related to **Stable Diffusion** in a single repo with unified requirements and...
- Working on `WSL2` with `CUDA`
- With latest versions of packages
- Running in global context Without `Anaconda`

## What's included?

- Original implementation  
- Memory-optimized implementation  
- Auxiliary models:  
  Multiple diffusers, face restoration, upsampling and detail enhancement  
- Basic GUI server  
- Full Web UI  

<br>

## Links & Credits

### Original work
- [Model Card](https://github.com/vladmandic/stable-diffusion/blob/main/MODEL-CARD.md)
- [Stable Diffusion Repository](https://github.com/CompVis/stable-diffusion) [[Readme]](https://github.com/vladmandic/stable-diffusion/blob/main/STABLE-DIFFUSION.md)
- [Stable Diffusion Announcement](https://stability.ai/blog/stable-diffusion-public-release)

### Enhancements
- [Memory Optimized Executor](https://github.com/basujindal/stable-diffusion)
- [Web-UI](https://github.com/hlky/stable-diffusion-webui) [[Readme]](https://github.com/vladmandic/stable-diffusion/blob/main/WEBUI.md)

### Auxiliary models
- [K-Diffusion](https://github.com/crowsonkb/k-diffusion)
- [Real ESRGAN](https://github.com/xinntao/Real-ESRGAN) ~101MB  
- [Latent Diffusion](https://github.com/devilismyfriend/latent-diffusion) ~2,056MB
- [GFPGAN](https://github.com/TencentARC/GFPGAN) ~550MB
- [Taming Transformers]() ~136MB

<br>

## Installation

### 1. Get correct PyTorch linked with CUDA

Make sure that **nVidia CUDA** is correctly installed and mark major/minor version:

> nvidia-smi  

    NVIDIA-SMI 510.73.05 Driver Version: 516.94 CUDA Version: 11.6

> head /usr/local/cuda/version.json  

    "cuda" : {
      "name" : "CUDA SDK",
      "version" : "11.6.1"
    },

Install **PyTorch** linked to *exact* version of **CUDA**:

> pip3 uninstall torch torchvision torchaudio  
> pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116  

Note that `cu116` at the end refers to `CUDA` **11.6** which should match `CUDA` installation on your system  
*Note*: **Stable-Diffusion** requires level **SM86** so older version of **CUDA** are likely insufficient  

Check functionality  

> python torchinfo.py  

    torch version: 1.12.1+cu116
    cuda available: True
    cuda version: 11.6
    cuda arch list: ['sm_37', 'sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86']
    device: NVIDIA GeForce RTX 3060

### 2. Install Dependencies

*Yes, there are tons...*  
*And all could be automated, but better to do it with some understanding...*  

**local packages**:
> pip install -e .  

**shared libraries**:
> pip install albumentations diffusers opencv-python pudb invisible-watermark imageio imageio-ffmpeg kornia   
> pip install pytorch-lightning omegaconf test-tube streamlit einops torch-fidelity transformers torchmetrics  
> pip install gradio pynvml basicsr facexlib  

**shared libraries from dev to get latest unpublished version**:
> pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers  
> pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip  
> pip install -e git+https://github.com/crowsonkb/k-diffusion/  

**local clones of some auxiliary models**:
> git clone --depth 1 https://github.com/devilismyfriend/latent-diffusion src/latent-diffusion  
> pip install -e src/latent-diffusion  

> git clone --depth 1 https://github.com/TencentARC/GFPGAN.git src/gfpgan  
> pip install -e src/gfpgan  

> git clone --depth 1 https://github.com/xinntao/Real-ESRGAN.git src/realesrgan  
> pip install -e src/realesrgan  

### 3. Download Model Weights

**stable-diffusion**:
- <https://huggingface.co/CompVis/stable-diffusion-v-1-4-original> -> `models/ldm/stable-diffusion-v1/model.ckpt`  
  Both `sd-v1-4.ckpt` and `sd-v1-4-full-ema.ckpt` are supported

**latent-diffusion**:
- <https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1> -> `src/latent-diffusion/experiments/project.yaml`
- <https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1> -> `src/latent-diffusion/experiments/pretrained_model/model.ckpt`

**gfpgan**:
- <https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth> -> `src/gfpgan/experiments/pretrained_models/`

**real-esrgan**:
- <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth> -> `src/realesrgan/experiments/pretrained_models/`
- <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth> -> `src/realesrgan/experiments/pretrained_models/`

<br>

## Run

### Web-UI with full options  
*Recommended*

      python webui.py
      Running on local URL:  http://localhost:7860/
### CLI Original
*Example*

      python scripts/txt2img.py --n_samples 2 --prompt "sketch of a female model riding a horse on a beach" --plms

### CLI Memory Optimized
*Example*

      python optimized/optimized_txt2img.py --n_samples 4 --turbo --prompt "sketch of a female model riding a horse on a beach"

### Basic GUI

      python optimized/txt2img_gradio.py  
      running on local URL:  http://127.0.0.1:7860/

<br>

## Random Notes

*Safe to Ignore*

### Enable PyTorch CUDA memory garbage collection

> export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

### Use FP16

> `scripts/txt2img.py`:`load_model_from_config` change from: `model.cuda()` to `model.cuda().half()`

### Reduce logging

```python
    from transformers import logging
    logging.set_verbosity_error()
```

![Example](https://github.com/vladmandic/stable-diffusion/raw/main/example.png)
