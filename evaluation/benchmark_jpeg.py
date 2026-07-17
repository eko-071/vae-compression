import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import numpy as np
import pandas as pd

from training.dataloader import get_dataloaders
from codec.compress import compress_with_jpeg
from codec.decompress import decompress_jpeg
from evaluation.metrics import compute_metrics


def benchmark_jpeg(config):
    dataset = config['dataset']
    num_images = config.get('num_images', 100)
    quality_min = config.get('quality_min', 1)
    quality_max = config.get('quality_max', 95)
    results_dir = config['results_dir']

    _, test_loader = get_dataloaders(dataset=dataset, batch_size=1)

    print(f"\nJPEG benchmark | {dataset} | {num_images} images | q{quality_min}-{quality_max}\n")

    results = []
    for quality in range(quality_min, quality_max + 1):
        psnr_scores, ssim_scores, size_scores = [], [], []

        for idx, (images, _) in enumerate(test_loader):
            if idx >= num_images:
                break

            jpeg_bytes, compressed_size = compress_with_jpeg(images, quality=quality)
            reconstructed = decompress_jpeg(jpeg_bytes)

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
            'model': 'JPEG',
            'dataset': dataset,
            'quality': quality,
            'avg_size_bytes': np.mean(size_scores),
            'avg_psnr': np.mean(psnr_scores),
            'avg_ssim': np.mean(ssim_scores),
        })

        if quality % 10 == 0 or quality == quality_max:
            print(f"  quality {quality:>3} | "
                  f"size {np.mean(size_scores):.0f}B | "
                  f"PSNR {np.mean(psnr_scores):.2f} | "
                  f"SSIM {np.mean(ssim_scores):.4f}")

    benchmark_dir = os.path.join(results_dir, 'benchmarks')
    os.makedirs(benchmark_dir, exist_ok=True)
    save_path = os.path.join(benchmark_dir, f"jpeg_{dataset}.csv")
    pd.DataFrame(results).to_csv(save_path, index=False)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    benchmark_jpeg(config)