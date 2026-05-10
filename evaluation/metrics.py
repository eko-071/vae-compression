import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_psnr(original, reconstructed):
    """
    This part is to compute PSNR, Peak Signal to Noise Ratio.
    Our inputs here are numpy arrays in range [0,1], shape (H, W) or (H, W, C).
    Returns PSNR in decibels, higher is better.
    Identical images should return infinity.
    """
    return peak_signal_noise_ratio(original, reconstructed, data_range=1.0)

def compute_ssim(original, reconstructed):
    """
    This part is to compute SSIM, Structural Similarity Index.
    Our inputs here are numpy arrays in range [0,1], shape (H, W) or (H, W, C).
    Returns SSIM in range [0, 1], higher is better.
    For color images, the channel_axis part tells skimage the last axis is channels.   
    """
    if original.ndim == 2:
        return structural_similarity(original, reconstructed, data_range=1.0)
    else:
        return structural_similarity(original, reconstructed, data_range=1.0, channel_axis=-1)
    
def compute_metrics(original, reconstructed):
    # Computes both and returns cleanly.
    return {
        'psnr': compute_psnr(original, reconstructed),
        'ssim': compute_ssim(original, reconstructed),
    }