# VAE Compression

A learned image compression system built with Variational Autoencoders, benchmarked against classical codecs (JPEG, PNG, zlib) across multiple datasets.

## Overview

Classical codecs like JPEG use fixed mathematical rules to compress images. This project explores whether neural networks can learn to compress images more effectively by learning the statistical structure of image data directly. We train a progression of models from a simple linear autoencoder up to a convolutional VAE, evaluate each against classical codecs using PSNR and SSIM metrics, and provide a CLI tool for compressing arbitrary images.

## Models

| Version | Model | Key Idea |
|---------|-------|----------|
| V1 | Linear Autoencoder | Baseline: mathematically equivalent to PCA |
| V2 | Convolutional Autoencoder | Exploits 2D spatial structure of images |
| V3 | Variational Autoencoder | Probabilistic latent space with KL regularization |
| V4 | Convolutional VAE | Combines V2 architecture with V3 variational bottleneck |

## Datasets

| Dataset | Resolution | Channels | Purpose |
|---------|-----------|----------|---------|
| MNIST | 28×28 | 1 | Development and baseline |
| CIFAR-10 | 32×32 | 3 | General natural images |
| CelebA | 64×64 | 3 | Domain-specific (faces) |

## Project Structure

```
.
├── checkpoints
├── cli.py
├── codec
├── data
├── evaluation
├── models
├── notebooks
├── README.md
├── requirements.txt
├── tests
└── training
    └─── configs
```

## Setup

```bash
git clone https://github.com/eko-071/vae-compression
cd vae-compression
pip install -r requirements.txt
```

Datasets download automatically when you first run training, except CelebA which requires manual setup: see `notebooks/01_explore_data.ipynb` for instructions.

## Training

```bash
python training/train.py --config training/configs/v1_linear_ae_mnist.yaml
python training/train.py --config training/configs/v2_convolutional_ae_mnist.yaml
python training/train.py --config training/configs/v3_vae_mnist.yaml
python training/train.py --config training/configs/v4_conv_vae_mnist.yaml
```

## CLI Usage

```bash
python cli.py --input <image> --model <model> --dataset <dataset> [options]
```

### Options

| Argument | Use |
| --- | --- |
| --input | Path to input image (required) |
| --model | v1_linear_ae, v2_convolutional_ae, v3_vae, v4_conv_vae (required) |
| --dataset | mnist, cifar10, celeba (required) |
| --latent_dim | Latent dimension of trained model (default: 32) |
| --jpeg_quality | JPEG quality for comparison, 1-95 (default: 75) |
| --output | Path to save side-by-side comparison image (optional) |

### Example

```bash
python cli.py --input photo.jpg --model v4_conv_vae --dataset cifar10 --latent_dim 128 --output comparison.png
```