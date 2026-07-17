import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import numpy as np
import pandas as pd
import torch

from training.dataloader import get_dataloaders
from training.model_loader import load_model_from_checkpoint
from codec.compress import compress_with_model
from codec.decompress import decompress_with_model
from evaluation.metrics import compute_metrics


def benchmark_model(config):
    dataset = config['dataset']
    model_name = config['model']
    save_dir = config['save_dir']
    results_dir = config['results_dir']
    num_images = config.get('num_images', 100)
    latent_dims = config.get('latent_dims') or [config.get('latent_dim', 32)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_dataloaders(dataset=dataset, batch_size=1)

    print(f"\nModel benchmark | {model_name} | {dataset} | {num_images} images\n")

    results = []
    for latent_dim in latent_dims:
        checkpoint = os.path.join(save_dir, f"{dataset}_ld{latent_dim}.pt")
        model = load_model_from_checkpoint(model_name, dataset, latent_dim, checkpoint)
        model = model.to(device)

        psnr_scores, ssim_scores, size_scores = [], [], []

        for idx, (images, _) in enumerate(test_loader):
            if idx >= num_images:
                break

            images = images.to(device)
            latent_bytes, compressed_size = compress_with_model(model, images)
            reconstructed = decompress_with_model(model, latent_bytes, latent_dim)

            orig_np = images.squeeze().cpu().numpy()
            recon_np = reconstructed.squeeze().cpu().numpy()

            if orig_np.ndim == 3:
                orig_np = np.transpose(orig_np, (1, 2, 0))
                recon_np = np.transpose(recon_np, (1, 2, 0))

            metrics = compute_metrics(orig_np, recon_np)
            psnr_scores.append(metrics['psnr'])
            ssim_scores.append(metrics['ssim'])
            size_scores.append(compressed_size)

        results.append({
            'model': model_name,
            'dataset': dataset,
            'latent_dim': latent_dim,
            'avg_size_bytes': np.mean(size_scores),
            'avg_psnr': np.mean(psnr_scores),
            'avg_ssim': np.mean(ssim_scores),
        })

        print(f"  ld={latent_dim:>4} | "
              f"size {np.mean(size_scores):.0f}B | "
              f"PSNR {np.mean(psnr_scores):.2f} | "
              f"SSIM {np.mean(ssim_scores):.4f}")

    benchmark_dir = os.path.join(results_dir, 'benchmarks')
    os.makedirs(benchmark_dir, exist_ok=True)
    save_path = os.path.join(benchmark_dir, f"{model_name}_{dataset}.csv")
    pd.DataFrame(results).to_csv(save_path, index=False)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    benchmark_model(config)