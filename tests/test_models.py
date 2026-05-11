import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import argparse
from training.dataloader import get_input_dim, DATASET_INFO
from training.model_loader import load_model, load_model_from_checkpoint

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dataset = config['dataset']
    model_name = config['model']
    latent_dim = config['latent_dim']
    checkpoint = os.path.join(config['save_dir'], f"{dataset}_ld{latent_dim}.pt")

    print(f"Testing {model_name} | {dataset} | latent_dim={latent_dim}")

    model = load_model(config)
    test_output_shape(model, dataset)
    test_output_range(model, dataset)
    test_checkpoint_loads(model, checkpoint)

    print("\nAll tests done.")