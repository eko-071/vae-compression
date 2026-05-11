import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
from training.dataloader import get_dataloaders
from training.model_loader import load_model_from_checkpoint
from codec.compress import compress_with_model, compress_with_jpeg, compress_with_png, compress_with_zlib, original_size
from codec.decompress import decompress_with_model, decompress_jpeg
from evaluation.metrics import compute_metrics


def run_codec_test(config):
    dataset = config['dataset']
    model_name = config['model']
    latent_dim = config['latent_dim']
    checkpoint = os.path.join(
        config['save_dir'],
        f"{dataset}_ld{latent_dim}.pt"
    )

    print(f"{model_name} | {dataset.upper()} | latent_dim={latent_dim}")

    model = load_model_from_checkpoint(model_name, dataset, latent_dim, checkpoint)

    _, test_loader = get_dataloaders(dataset=dataset, data_dir='./data', batch_size=1)
    image, _ = next(iter(test_loader))

    raw_size = original_size(image)

    latent_bytes, model_size = compress_with_model(model, image)
    reconstruction = decompress_with_model(model, latent_bytes, model.latent_dim)

    jpeg_bytes, jpeg_size = compress_with_jpeg(image, quality=75)
    jpeg_recon = decompress_jpeg(jpeg_bytes)

    _, png_size = compress_with_png(image)
    _, zlib_size = compress_with_zlib(image)

    def to_np(t):
        return t.squeeze(0).permute(1, 2, 0).detach().numpy()

    orig_np = to_np(image)
    model_metrics = compute_metrics(orig_np, to_np(reconstruction))
    jpeg_metrics  = compute_metrics(orig_np, to_np(jpeg_recon))

    print(f"Original size: {raw_size} bytes")
    print(f"Model compressed: {model_size} bytes | PSNR {model_metrics['psnr']:.2f} dB | SSIM {model_metrics['ssim']:.4f}")
    print(f"JPEG compressed: {jpeg_size} bytes | PSNR {jpeg_metrics['psnr']:.2f} dB | SSIM {jpeg_metrics['ssim']:.4f}")
    print(f"PNG compressed: {png_size} bytes | (lossless, no reconstruction metrics)")
    print(f"zlib compressed: {zlib_size} bytes | (lossless, no reconstruction metrics)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    run_codec_test(config)