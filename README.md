# CompVis Stable-Diffusion

Custom installation...

- Working on `WSL2` with `CUDA`
- With latest versions of packages
- Running in global context Without `Anaconda`

Contains 

- Original implementation
- Memory-optimized impolementation
- GUI server

## Sources

- Original: <https://github.com/CompVis/stable-diffusion>
- Optimized: <https://github.com/basujindal/stable-diffusion>

## Installation

### 1. Get correct PyTorch linked with CUDA

Make sure that **nVidia CUDA** is correctly installed and mark version:

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

Check functionality  

*Note*: **Stable-Diffusion** requires level **SM86** so older version of **CUDA** are likely insufficient  

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

> mkdir models/ldm/stable-diffusion-v1  
> ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt 

## Run

### Original

> python scripts/txt2img.py --n_samples 2 --prompt "sketch of a female model riding a horse on a beach" --plms

    Global seed set to 42
    Loading model from models/ldm/stable-diffusion-v1/model.ckpt
    Global Step: 470000
    LatentDiffusion: Running in eps-prediction mode
    DiffusionWrapper has 859.52 M params.
    making attention of type 'vanilla' with 512 in_channels
    Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
    making attention of type 'vanilla' with 512 in_channels
    Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...
    Sampling:
    Running PLMS Sampling with 50 timesteps
    Your samples are ready and waiting for you here: outputs/txt2img-samples

![Example](https://github.com/vladmandic/stable-diffusion/raw/main/example.png)

### Memory Optimized

> python optimizedSD/optimized_txt2img.py --n_samples 2 --prompt "sketch of a female model riding a horse on a beach"

### Using GUI

> python optimizedSD/txt2img_gradio.py

    Loading model from models/ldm/stable-diffusion-v1/model.ckpt
    Global Step: 470000
    UNet: Running in eps-prediction mode
    CondStage: Running in eps-prediction mode
    FirstStage: Running in eps-prediction mode
    making attention of type 'vanilla' with 512 in_channels
    Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
    making attention of type 'vanilla' with 512 in_channels
    Running on local URL:  http://127.0.0.1:7860/

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
