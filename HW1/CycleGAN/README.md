# CycleGAN Implementation

This directory contains a complete PyTorch implementation of CycleGAN (Cycle-Consistent Generative Adversarial Networks) for unpaired image-to-image translation.

## Overview

CycleGAN enables image translation between two domains without paired training data. The model learns mappings G: X → Y and F: Y → X along with adversarial discriminators DY and DX. The key innovation is the cycle consistency loss that enforces F(G(X)) ≈ X and G(F(Y)) ≈ Y.

## Features

- **Original CycleGAN**: Support for unpaired image datasets
- **MNIST Support**: Integrated with custom `Mnist_Dataloader.py` for MNIST digit translation
- **Configurable**: Easy configuration through `config.py`
- **Example Scripts**: Ready-to-run examples for different datasets

## Architecture

### Generator
- **Architecture**: c7s1-64, d128, d256, R256×6/9, u128, u64, c7s1-3
- **Components**: 
  - c7s1-k: 7×7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1
  - dk: 3×3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2
  - Rk: Residual block with two 3×3 convolutional layers
  - uk: 3×3 fractional-strided-Convolution-InstanceNorm-ReLU layer with k filters and stride 1/2

### Discriminator (PatchGAN)
- **Architecture**: C64-C128-C256-C512
- **Components**:
  - Ck: 4×4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2
  - Uses LeakyReLU with slope 0.2
  - Outputs patch-wise classification

## Files Description

1. **CycleGan_Generator.py**: Generator network implementation
2. **CycleGan_Discriminator.py**: Discriminator network implementation
3. **dataset.py**: Dataset loader for unpaired image datasets and MNIST
4. **utils.py**: Utility functions for saving/loading checkpoints and samples
5. **train.py**: Main training script with CycleGANTrainer class
6. **test.py**: Inference and testing script
7. **config.py**: Configuration parameters
8. **mnist_example.py**: Example script for MNIST dataset usage

## MNIST Integration

The implementation now includes support for MNIST dataset using the custom `Mnist_Dataloader.py`:

### Features:
- **Domain Separation**: Configurable digit groups (default: even vs odd digits)
- **Automatic Loading**: Uses the custom MNIST loader for data handling
- **Flexible Configuration**: Easy to modify digit domains in `config.py`

### Usage:
1. Set `USE_MNIST = True` in `config.py`
2. Configure `MNIST_DATA_PATH` to point to your MNIST dataset
3. Optionally modify `MNIST_DOMAIN_A_DIGITS` and `MNIST_DOMAIN_B_DIGITS`
4. Run training: `python train.py`

### Example:
```python
# Run the MNIST example
python mnist_example.py
```

## Loss Functions

The total loss consists of:

1. **Adversarial Loss**: 
   ```
   L_GAN(G, D_Y, X, Y) = E_y[log D_Y(y)] + E_x[log(1 - D_Y(G(x)))]
   ```

2. **Cycle Consistency Loss**:
   ```
   L_cyc(G, F) = E_x[||F(G(x)) - x||_1] + E_y[||G(F(y)) - y||_1]
   ```

3. **Identity Loss** (optional):
   ```
   L_identity(G, F) = E_y[||G(y) - y||_1] + E_x[||F(x) - x||_1]
   ```

**Total Loss**:
```
L(G, F, D_X, D_Y) = L_GAN(G, D_Y, X, Y) + L_GAN(F, D_X, Y, X) + λ_cyc * L_cyc(G, F) + λ_identity * L_identity(G, F)
```

## Setup and Usage

### Prerequisites
```bash
pip install torch torchvision tqdm matplotlib pillow
```

### Dataset Preparation
1. Create the following directory structure:
```
data/
├── trainA/     # Domain A training images
├── trainB/     # Domain B training images  
├── testA/      # Domain A test images
└── testB/      # Domain B test images
```

2. Popular datasets:
   - Horse2Zebra: https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset
   - Apple2Orange, Summer2Winter, etc.

### Training

1. **Configure parameters** in `config.py`:
   ```python
   TRAIN_DIR_A = "data/trainA"  # Set your data paths
   TRAIN_DIR_B = "data/trainB"
   NUM_EPOCHS = 100
   LEARNING_RATE = 2e-4
   LAMBDA_CYCLE = 10
   ```

2. **Start training**:
   ```bash
   python train.py
   ```

3. **Monitor training**:
   - Checkpoints saved in `checkpoints/` directory
   - Sample translations saved in `samples/` directory
   - Training progress displayed with tqdm

### Testing/Inference

1. **Test model architecture**:
   ```bash
   python test.py
   ```

2. **Generate translations** (after training):
   ```python
   from test import translate_image, load_model
   
   generator = load_model("checkpoints/gen_h.pth.tar", device)
   output_image = translate_image(generator, "input.jpg", "output.jpg")
   ```

## Training Tips

1. **Batch Size**: Use batch_size=1 for best results (as in original paper)
2. **Learning Rate**: Start with 2e-4, decay after 100 epochs
3. **Lambda Values**: 
   - λ_cycle = 10 (cycle consistency weight)
   - λ_identity = 0.5 (identity loss weight, optional)
4. **Residual Blocks**: Use 6 for 128×128 images, 9 for 256×256 images
5. **Training Time**: ~4-6 hours for 100 epochs on GPU

## Key Features

- **Unpaired Training**: No need for paired datasets
- **Cycle Consistency**: Ensures meaningful translations
- **PatchGAN Discriminator**: Better texture discrimination
- **Identity Loss**: Preserves color composition
- **Flexible Architecture**: Configurable residual blocks
- **Mixed Precision**: Automatic mixed precision support
- **Checkpoint System**: Resume training from saved states

## Results

The model typically shows good results after 50-100 epochs:
- Learns to translate key objects (e.g., horse ↔ zebra stripes)
- Preserves spatial structure and poses
- May need longer training for background quality

## References

1. [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
2. [CycleGAN Project Page](https://junyanz.github.io/CycleGAN/)
3. [PyTorch Implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## Implementation Notes

This implementation is based on:
- The original CycleGAN paper by Zhu et al.
- Medium article by Denaya with practical PyTorch implementation
- Best practices from the community

The code is designed to be educational and easily modifiable for different domains and datasets.
