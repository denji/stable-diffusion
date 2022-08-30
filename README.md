# CompVis Stable-Diffusion

Custom installation...

- Working on `WSL2` with `CUDA`
- With latest versions of packages
- Running in global context Without `Anaconda`

Contains 

- Original implementation
- Memory-optimized impolementation
- GUI server

## Links & Credits

- [Stable Diffusion Announcement](https://stability.ai/blog/stable-diffusion-public-release)
- [Stable Diffusion Repository](https://github.com/CompVis/stable-diffusion)
- [Original Notes](https://github.com/vladmandic/stable-diffusion/blob/main/STABLE-DIFFUSION.md)
- [Model Card](https://github.com/vladmandic/stable-diffusion/blob/main/MODEL-CARD.md)
- [Memory Optimized Executor](https://github.com/basujindal/stable-diffusion)

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

> pip install albumentations diffusers opencv-python pudb invisible-watermark imageio imageio-ffmpeg kornia  
> pip install pytorch-lightning omegaconf test-tube streamlit einops torch-fidelity transformers torchmetrics gradio  
> pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers  
> pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip  
> pip install -e .  

### 3. Download Model Weights

Download from: <https://huggingface.co/CompVis/stable-diffusion-v-1-4-original>  
Both `sd-v1-4.ckpt` and `sd-v1-4-full-ema.ckpt` are supported

> ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt 

## Run

### Original

> python scripts/txt2img.py --n_samples 2 --prompt "sketch of a female model riding a horse on a beach" --plms

### Memory Optimized

> python optimizedSD/optimized_txt2img.py --n_samples 4 --turbo --prompt "sketch of a female model riding a horse on a beach"

    loading model: models/ldm/stable-diffusion-v1/model.ckpt
    global step: 470000
    ddpm: UNet mode: eps
    ddpm: CondStage mode: eps
    ddpm: FirstStage mode: eps
    working with z of shape (1, 4, 32, 32) = 4096 dimensions.
    attention type: 'vanilla' channels: 512
    params: {'prompt': 'sketch of a female model riding a horse on a beach', 'outdir': 'outputs/txt2img-samples', 'skip_grid': False, 'skip_save': False, 'ddim_steps': 50, 'fixed_code': False, 'ddim_eta': 0.0, 'n_iter': 1, 'H': 512, 'W': 512, 'C': 4, 'f': 8, 'n_samples': 4, 'n_rows': 0, 'scale': 7.5, 'device': 'cuda', 'from_file': None, 'seed': 162453, 'unet_bs': 1, 'turbo': True, 'precision': 'autocast', 'format': 'png'}
    resolution: 512 x 512 x 4
    sampler: PLMS
    iterations: 1
    samples per iteration: 4
    sample timesteps: 50
    start iteration: 0
    seeds used:  [162453, 162454, 162455, 162456]
    progress: 100% | 50/50 [00:47<00:00,  1.05it/s]
    saving images
    gpu memory allocated:  3907.0 MB
    export folder: outputs/txt2img-samples/sketch_of_a_female_model_riding_a_horse_on_a_beach
    wall: 61.1 sec load: 8.6 sec sample: 13.1 sec

![Example](https://github.com/vladmandic/stable-diffusion/raw/main/example.png)

### Using GUI

> python optimizedSD/txt2img_gradio.py

    loading model from models/ldm/stable-diffusion-v1/model.ckpt
    ddpm: UNet mode: eps
    ddpm: CondStage mode: eps
    ddpm: FirstStage mode: eps
    attention type: 'vanilla' channels: 512
    running on local URL:  http://127.0.0.1:7860/

## Additional Notes

### Enable PyTorch CUDA memory garbage collection

> export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

### Use FP16

> `scripts/txt2img.py`:`load_model_from_config` change from: `model.cuda()` to `model.cuda().half()`

### Reduce logging

```python
    from transformers import logging
    logging.set_verbosity_error()
```
