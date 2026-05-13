import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import numpy as np
import pandas as pd
import torch

from training.dataloader import get_dataloaders
from training.model_loader import load_model

from codec.compress import (compress_with_model)
from codec.decompress import(decompress_with_model)

from evaluation.metrics import compute_metrics

def benchmark_model(config):
    dataset = config['dataset']
    latent_dim = config['latent_dim']
    model_name = config['model']
    save_dir = config['save_dir']
    results_dir = config['results_dir']

    num_images = config.get('num_images', 100)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    train_loader, test_loader = get_dataloaders(
        dataset=dataset,
        batch_size=1
    )
    model = load_model(config).to(device)
    checkpoint_path = os.path.join(
        save_dir,
        f"{dataset}_ld{latent_dim}.pt"
    )
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )
    model.eval()
    psnr_scores = []

    ssim_scores = []

    size_scores = []
    print("\nStarting model benchmark...")

    print(f"Model: {model_name}")

    print(f"Dataset: {dataset}")

    print(f"Latent dimension: {latent_dim}")

    print(f"Images: {num_images}\n")
    
    for idx, (images, _) in enumerate(test_loader):
        if idx >= num_images:
            break
        images = images.to(device)
        latent_bytes, compressed_size = compress_with_model(
            model,
            images
        )
        reconstructed = decompress_with_model(
            model,
            latent_bytes,
            latent_dim
        )
        original_np = images.squeeze().cpu().numpy()

        reconstructed_np = reconstructed.squeeze().cpu().numpy()    
        if original_np.ndim == 3:

            original_np = np.transpose(
                original_np,
                (1, 2, 0)
            )

            reconstructed_np = np.transpose(
                reconstructed_np,
                (1, 2, 0)
            )
        metrics = compute_metrics(
            original_np,
            reconstructed_np
        )
        psnr_scores.append(metrics['psnr'])

        ssim_scores.append(metrics['ssim'])

        size_scores.append(compressed_size)
    avg_psnr = np.mean(psnr_scores)

    avg_ssim = np.mean(ssim_scores)

    avg_size = np.mean(size_scores)
    results = pd.DataFrame([{
        'model': model_name,
        'dataset': dataset,
        'latent_dim': latent_dim,
        'avg_size_bytes': avg_size,
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim
    }])
    benchmark_dir = os.path.join(
        results_dir,
        "benchmarks"
    )

    os.makedirs(benchmark_dir, exist_ok=True)
    save_path = os.path.join(
        benchmark_dir,
        f"{model_name}_{dataset}_ld{latent_dim}.csv"
    )

    results.to_csv(
        save_path,
        index=False
    )
    print("Benchmark complete.")

    print(f"Saved results to {save_path}")
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        type=str,
        required=True
    )

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    benchmark_model(config)