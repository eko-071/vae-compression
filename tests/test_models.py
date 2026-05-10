import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from training.dataloader import get_input_dim, DATASET_INFO
from models.v1_linear_ae import LinearAutoencoder

def test_output_shape(model, dataset):
    info = DATASET_INFO[dataset]
    C, H, W = info['channels'], info['height'], info['width']
    batch = torch.randn(4, C, H, W)

    reconstruction, latent = model(batch)

    assert reconstruction.shape == batch.shape, f"Reconstruction shape {reconstruction.shape} doesn't match input shape {batch.shape}"
    assert latent.shape == (4, model.latent_dim), f"Latent shape {latent.shape} is wrong, expected (4, {model.latent_dim})"

    print(f"Shape test passed: input {tuple(batch.shape)} -> latent {tuple(latent.shape)} -> output {tuple(reconstruction.shape)}")

def test_output_range(model, dataset):
    info = DATASET_INFO[dataset]
    C, H, W = info['channels'], info['height'], info['width']
    batch = torch.randn(4, C, H, W)

    reconstruction, _ = model(batch)

    assert reconstruction.min().item() >= 0.0, "Output below 0"
    assert reconstruction.max().item() <= 1.0, "Output above 1"

    print(f"Range test passed: output in range [{reconstruction.min().item():.4f}, {reconstruction.max().item():.4f}]")

def test_checkpoint_loads(model, checkpoint_path):
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model.eval()
        print(f"Checkpoint test passed: loaded {checkpoint_path}")
    except Exception as e:
        print(f"Checkpoint test failed: {e}")

if __name__ == "__main__":
    print("Testing V1: Linear Autoencoder")

    print()
    print("Testing MNIST:")
    info = DATASET_INFO['mnist']
    model = LinearAutoencoder(
        input_dim=get_input_dim('mnist'),
        latent_dim=32,
        channels=info['channels'],
        height=info['height'],
        width=info['width']
    )

    test_output_shape(model, 'mnist')
    test_output_range(model, 'mnist')
    test_checkpoint_loads(model, './checkpoints/v1_linear_ae/mnist_ld32.pt')
    
    print()
    print("Testing CIFAR-10:")
    info = DATASET_INFO['cifar10']
    model = LinearAutoencoder(
        input_dim=get_input_dim('cifar10'),
        latent_dim=32,
        channels=info['channels'],
        height=info['height'],
        width=info['width']
    )

    test_output_shape(model, 'cifar10')
    test_output_range(model, 'cifar10')
    test_checkpoint_loads(model, './checkpoints/v1_linear_ae/cifar10_ld32.pt')

    print()
    print("All tests done.")