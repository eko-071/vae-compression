import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import argparse
import yaml
import numpy as np

from training.dataloader import get_dataloaders
from codec.compress import (
    compress_with_jpeg
)
from codec.decompress import (decompress_jpeg)

from evaluation.metrics import compute_metrics

def benchmark_jpeg(config):
    dataset = config['dataset']

    num_images = config.get('num_images', 100)

    quality_min = config.get('quality_min', 1)
    quality_max = config.get('quality_max', 95)

    results_dir = config['results_dir']
    train_loader, test_loader = get_dataloaders(
    dataset=dataset,
    batch_size=1
    )
    results=[]
    print("\nStarting JPEG benchmark...")
    print(f"Dataset: {dataset}")
    print(f"Images: {num_images}")
    print(f"Qualities: {quality_min}-{quality_max}\n")
    for quality in range(quality_min, quality_max + 1):
        psnr_scores = []
        ssim_scores = []
        size_scores = []
        for idx, (images, _) in enumerate(test_loader):
            if idx >= num_images:
                break
            jpeg_bytes, compressed_size = compress_with_jpeg(
                images,
                quality=quality
            )
            reconstructed = decompress_jpeg(jpeg_bytes)
            original_np = images.squeeze().cpu().numpy()
            reconstructed_np = reconstructed.squeeze().cpu().numpy()
            if original_np.ndim == 3:
                original_np = np.transpose(original_np, (1, 2, 0))
                reconstructed_np = np.transpose(reconstructed_np, (1, 2, 0))
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
        results.append({
            'dataset': dataset,
            'quality': quality,
            'avg_size_bytes': avg_size,
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim
        })
        if quality % 10 == 0 or quality == quality_max:
            print(f"Completed quality {quality}")
    df = pd.DataFrame(results)

    benchmark_dir = os.path.join(results_dir, "benchmarks")

    os.makedirs(benchmark_dir, exist_ok=True)

    save_path = os.path.join(
        benchmark_dir,
        f"jpeg_{dataset}.csv"
    )

    df.to_csv(save_path, index=False)

    print(f"\nSaved results to {save_path}")

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

    benchmark_jpeg(config)