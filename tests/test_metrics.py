import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from evaluation.metrics import compute_metrics

if __name__ == "__main__":
    img = np.random.rand(28, 28).astype(np.float32)
    metrics = compute_metrics(img, img)
    print("Identical images:")
    print(f"PSNR: {metrics['psnr']}")
    print(f"SSIM: {metrics['ssim']:.4f}\n")

    noisy = np.clip(img + np.random.normal(0, 0.01, img.shape), 0, 1).astype(np.float32)
    metrics = compute_metrics(img, noisy)
    print("Little bit noisy:")
    print(f"PSNR: {metrics['psnr']:.4f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}\n")

    noisy = np.clip(img + np.random.normal(0, 0.3, img.shape), 0, 1).astype(np.float32)
    metrics = compute_metrics(img, noisy)
    print("Very noisy:")
    print(f"PSNR: {metrics['psnr']:.4f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}\n")

    img_color = np.random.rand(64, 64, 3).astype(np.float32)
    metrics = compute_metrics(img_color, img_color)
    print("Identical color images:")
    print(f"PSNR: {metrics['psnr']}")
    print(f"SSIM: {metrics['ssim']:.4f}")
