import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import numpy as np
from PIL import Image
from tabulate import tabulate

from training.dataloader import DATASET_INFO, get_input_dim
from training.model_loader import load_model_from_checkpoint
from codec.compress import compress_with_model, compress_with_jpeg, compress_with_png, compress_with_zlib, original_size
from codec.decompress import decompress_with_model, decompress_jpeg
from evaluation.metrics import compute_metrics

AVAILABLE_MODELS = {
    'v1_linear_ae':      'checkpoints/v1_linear_ae',
    'v2_convolutional_ae': 'checkpoints/v2_convolutional_ae',
    'v3_vae':            'checkpoints/v3_vae',
    'v4_conv_vae':       'checkpoints/v4_conv_vae',
}


def load_image(image_path, dataset):
    """
    Load an image from disk and resize/normalize it to match
    the target dataset's expected dimensions.
    Returns a tensor of shape (1, C, H, W) in [0, 1].
    """
    info = DATASET_INFO[dataset]
    H, W, C = info['height'], info['width'], info['channels']

    img = Image.open(image_path)

    if C == 1:
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    img = img.resize((W, H), Image.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0

    if img_np.ndim == 2:
        img_np = img_np[:, :, np.newaxis]

    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).contiguous()
    return img_tensor


def find_checkpoint(model_name, dataset, latent_dim):
    """
    Looks for a checkpoint matching the model, dataset, and latent_dim.
    """
    checkpoint_dir = AVAILABLE_MODELS[model_name]
    filename = f"{dataset}_ld{latent_dim}.pt"
    path = os.path.join(checkpoint_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No checkpoint found at {path}\n"
            f"Make sure you have trained this model first:\n"
            f"  python training/train.py --config training/configs/{model_name}_{dataset}.yaml"
        )

    return path


def save_output(original, reconstruction, output_path, dataset):
    """
    Save original and reconstructed images side by side.
    """
    info = DATASET_INFO[dataset]

    def to_pil(tensor):
        img_np = tensor.squeeze(0).permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        if info['channels'] == 1:
            return Image.fromarray(img_np.squeeze(-1), mode='L')
        return Image.fromarray(img_np, mode='RGB')

    orig_pil = to_pil(original)
    recon_pil = to_pil(reconstruction)

    combined = Image.new(
        'RGB' if info['channels'] == 3 else 'L',
        (info['width'] * 2, info['height'])
    )
    combined.paste(orig_pil, (0, 0))
    combined.paste(recon_pil, (info['width'], 0))
    combined.save(output_path)
    print(f"Saved comparison image to {output_path}")

def run(args):
    if args.model not in AVAILABLE_MODELS:
        print(f"Unknown model: {args.model}")
        print(f"Available: {', '.join(AVAILABLE_MODELS.keys())}")
        sys.exit(1)

    if args.dataset not in DATASET_INFO:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {', '.join(DATASET_INFO.keys())}")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Image not found: {args.input}")
        sys.exit(1)

    print(f"\nLoading image: {args.input}")
    image = load_image(args.input, args.dataset)

    checkpoint = find_checkpoint(args.model, args.dataset, args.latent_dim)
    print(f"Loading model: {args.model} | {args.dataset} | latent_dim={args.latent_dim}")
    model = load_model_from_checkpoint(
        args.model, args.dataset, args.latent_dim, checkpoint
    )

    print("Compressing...")
    latent_bytes, model_size = compress_with_model(model, image)
    reconstruction = decompress_with_model(model, latent_bytes, model.latent_dim)

    jpeg_bytes, jpeg_size = compress_with_jpeg(image, quality=args.jpeg_quality)
    jpeg_recon = decompress_jpeg(jpeg_bytes)

    _, png_size = compress_with_png(image)
    _, zlib_size = compress_with_zlib(image)
    raw_size = original_size(image)

    def to_np(t):
        return t.squeeze(0).permute(1, 2, 0).detach().numpy()

    model_metrics = compute_metrics(to_np(image), to_np(reconstruction))
    jpeg_metrics = compute_metrics(to_np(image), to_np(jpeg_recon))

    model_ratio = raw_size / model_size
    jpeg_ratio = raw_size / jpeg_size

    table = [
        ["Original",
         f"{raw_size} bytes",
         f"{raw_size / 1024:.2f} KB",
         "-", "-", "-"],
        [f"Model ({args.model})",
         f"{model_size} bytes",
         f"{model_size / 1024:.2f} KB",
         f"{model_ratio:.1f}:1",
         f"{model_metrics['psnr']:.2f} dB",
         f"{model_metrics['ssim']:.4f}"],
        [f"JPEG (q={args.jpeg_quality})",
         f"{jpeg_size} bytes",
         f"{jpeg_size / 1024:.2f} KB",
         f"{jpeg_ratio:.1f}:1",
         f"{jpeg_metrics['psnr']:.2f} dB",
         f"{jpeg_metrics['ssim']:.4f}"],
        ["PNG (lossless)",
         f"{png_size} bytes",
         f"{png_size / 1024:.2f} KB",
         f"{raw_size / png_size:.1f}:1",
         "-", "-"],
        ["zlib (lossless)",
         f"{zlib_size} bytes",
         f"{zlib_size / 1024:.2f} KB",
         f"{raw_size / zlib_size:.1f}:1",
         "-", "-"],
    ]

    headers = ["Method", "Size", "Size (KB)", "Ratio", "PSNR", "SSIM"]
    print(f"\n{tabulate(table, headers=headers, tablefmt='rounded_outline')}\n")

    if model_metrics['ssim'] > jpeg_metrics['ssim']:
        print(f"Model wins on SSIM at this compression level.")
    elif model_metrics['psnr'] > jpeg_metrics['psnr']:
        print(f"Model wins on PSNR at this compression level.")
    else:
        print(f"JPEG wins at this compression level.")
        print(f"Try a higher latent_dim or a more advanced model.")

    if args.output:
        save_output(image, reconstruction, args.output, args.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='VAE Compression: compress an image and compare against classical codecs'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list(AVAILABLE_MODELS.keys()),
        help='Model to use for compression'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(DATASET_INFO.keys()),
        help='Dataset the model was trained on'
    )
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=32,
        help='Latent dimension of the trained model (default: 32)'
    )
    parser.add_argument(
        '--jpeg_quality',
        type=int,
        default=75,
        help='JPEG quality for comparison 1-95 (default: 75)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional path to save side-by-side comparison image'
    )

    args = parser.parse_args()
    run(args)