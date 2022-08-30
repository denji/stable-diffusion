# CompVis Stable-Diffusion

Custom installation...

- Working on `WSL2`
- With `CUDA` 11.6
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

### Get correct PyTorch linked with CUDA

    pip3 uninstall torch torchvision torchaudio
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

    python ~/dev/tfjs-utils/src/torchinfo.py

    torch version: 1.12.1+cu116
    cuda available: True
    cuda version: 11.6
    cuda arch list: ['sm_37', 'sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86']
    device: NVIDIA GeForce RTX 3060

### Install Stable Diffusion

    git clone --depth 1 https://github.com/CompVis/stable-diffusion stable-diffusion
    git clone --depth 1 https://github.com/basujindal/stable-diffusion optimized
    mv optimized/optimizedSD stable-diffusion/
    cd stable-diffusion
    rm -rf data
    pip install albumentations diffusers opencv-python pudb invisible-watermark imageio imageio-ffmpeg kornia
    pip install pytorch-lightning omegaconf test-tube streamlit einops torch-fidelity transformers torchmetrics gradio
    pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
    pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
    pip install -e .

### Download Weights

> <https://huggingface.co/CompVis/stable-diffusion-v-1-4-original>

    mkdir models/ldm/stable-diffusion-v1
    ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt 

## Run

### Original

    python scripts/txt2img.py --n_samples 2 --prompt "sketch of a female model riding a horse on a beach" --plms

![Example](https://github.com/vladmandic/stable-diffusion/raw/main/example.png)

### Optimized

    python optimizedSD/optimized_txt2img.py --n_samples 2 --prompt "sketch of a female model riding a horse on a beach"

### Using GUI

    python optimizedSD/txt2img_gradio.py

## Additional Notes

### Enable PyTorch CUDA memory garbage collection

    export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

### Use FP16

    scripts/txt2img.py:load_model_from_config` change from: `model.cuda()` to `model.cuda().half()

### Reduce logging

    from transformers import logging
    logging.set_verbosity_error()
