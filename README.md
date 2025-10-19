# paper-impl
This repository is inspired by the original authors of the implemented papers and builds upon insights from the research community.

# Paper-Impl: Research Paper Implementations

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview

**Paper-Impl** is a meticulously organized repository containing reimplementations of seminal papers in computer vision and deep learning. Each implementation focuses on:

- **Readability**: Clean, well-documented code that mirrors the original papers
- **Reproducibility**: Faithful recreations of experimental setups and results
- **Modularity**: Reusable components that can be mixed and matched
- **Educational Value**: Detailed comments explaining key concepts and design choices

## 🎯 Implemented Papers

### Self-Supervised Learning
- **[SimCLR](src/papers/simclr/)** (Chen et al., 2020) - Contrastive learning framework with novel data augmentation strategies
  - [Model Code](src/papers/simclr/model.py) | [Loss Function](src/papers/simclr/loss.py) | [Training Script](src/papers/simclr/train.py)
- **[MoCo](src/papers/moco/)** (He et al., 2020) - Momentum contrast for unsupervised visual representation learning
  - [Model Code](src/papers/moco/model.py) | [Training Script](src/papers/moco/train.py)

### Architectures  
- **[ResNet](src/papers/resnet/)** (He et al., 2015) - Residual learning framework for deep networks
  - [ResNet Implementation](src/papers/resnet/resnet.py) | [Bottleneck Block](src/papers/resnet/bottleneck.py) | [Config](configs/papers/resnet.yaml)
- **[Vision Transformer](src/papers/vit/)** (Dosovitskiy et al., 2020) - Transformer architecture for image recognition
  - [ViT Model](src/papers/vit/vit.py) | [Attention Module](src/papers/vit/attention.py) | [Embeddings](src/papers/vit/embeddings.py)

### Generative Models
- **[DDPM](src/papers/ddpm/)** (Ho et al., 2020) - Denoising diffusion probabilistic models
  - [U-Net](src/papers/ddpm/unet.py) | [Diffusion Process](src/papers/ddpm/diffusion.py) | [Scheduler](src/papers/ddpm/scheduler.py)
- **[Stable Diffusion](src/papers/stable_diffusion/)** (Rombach et al., 2022) - High-resolution image synthesis with latent diffusion
  - [VAE](src/papers/stable_diffusion/vae.py) | [CLIP Text Encoder](src/papers/stable_diffusion/clip_text.py)

### Multimodal Learning
- **[CLIP](src/papers/clip/)** (Radford et al., 2021) - Connecting text and images with contrastive pre-training
  - [CLIP Model](src/papers/clip/model.py) | [Tokenizer](src/papers/clip/tokenizer.py) | [Config](configs/papers/clip.yaml)

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/yourusername/paper-impl
cd paper-impl

# Install with pip
pip install -r requirements.txt

# Or install in development mode
pip install -e .
