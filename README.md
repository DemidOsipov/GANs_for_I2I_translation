# GANs for Unpaired Image-to-Image Translation

Created during the YSDA course on Generative AI. This project implements **Generative Adversarial Networks (GANs)** for unpaired image-to-image translation using L1 regularization as an alternative to cycle consistency loss. The work demonstrates that simpler GAN architectures can effectively perform domain transfer between the MNIST and USPS handwritten digit datasets.

## Overview

Traditional unpaired I2I translation methods like CycleGAN require complex architectures with dual generators and discriminators plus cycle consistency loss. This implementation shows that **L1 regularization** between input and generated images can achieve comparable results with significantly reduced complexity.

## Key Features

- **Simplified Architecture**: Single generator-discriminator pair instead of dual networks
- **L1 Regularization**: `||G(x) - x||₁` loss for maintaining structural similarity
- **Multiple GAN Objectives**: Support for Vanilla GAN, Non-saturating GAN, WGAN, and LSGAN losses
- **Patch Discriminator**: Efficient discriminator focusing on local image patches
- **Domain Transfer**: MNIST → USPS handwritten digit translation

## Technical Implementation

### Architecture Components
- **Generator**: 7-layer convolutional network with ReLU activations
- **Discriminator**: PatchGAN discriminator with InstanceNorm and LeakyReLU
- **Loss Function**: Combined adversarial loss + L1 regularization

### Training Strategy
```python
Loss_Generator = Loss_GAN + λ * ||G(x) - x||₁
Loss_Discriminator = (Loss_real + Loss_fake) / 2
```

### Dataset Processing
- **Input**: MNIST and USPS datasets resized to 16×16 pixels
- **Preprocessing**: Normalization to [-1, 1] range
- **Batch Size**: 64 samples for efficient training

## Results

The model successfully performs unpaired domain translation:
- **Qualitative**: Generated USPS-style digits visually similar to target domain
- **Efficiency**: Requires 50% fewer parameters than CycleGAN approach
- **Training Speed**: Fast convergence due to simple loss function

## Key Findings

1. **L1 regularization** effectively preserves digit structure during domain transfer
2. **Simplified architecture** achieves comparable quality to complex cycle-consistent models
3. **PatchGAN discriminator** provides efficient training for small image resolutions
4. The approach works well for **simple domains** like handwritten digits

## Requirements

- PyTorch 1.0+
- torchvision
- numpy
- matplotlib
- tqdm

## Usage

The notebook provides a complete training pipeline:
1. Dataset loading and preprocessing
2. Model architecture definition
3. Training loop with loss visualization
4. Qualitative evaluation on test data


