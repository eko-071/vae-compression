import io
import torch
import numpy as np
from PIL import Image

def decompress_with_model(model, latent_bytes, latent_dim):
    """
    Reconstructs an image from raw latent bytes.
    Returns image as tensor (1, C, H, W) in [0, 1].
    """
    latent_np = np.frombuffer(latent_bytes, dtype=np.float32).reshape(1, latent_dim)
    z = torch.tensor(latent_np)

    model.eval()
    with torch.no_grad():
        reconstruction = model.decode(z)

    return reconstruction

def decompress_jpeg(jpeg_bytes):
    """
    Decompresses JPEG bytes back into a tensor.
    """
    buffer = io.BytesIO(jpeg_bytes)
    img_pil = Image.open(buffer)
    img_np = np.array(img_pil).astype(np.float32) / 255.0

    if img_np.ndim == 2:
        img_np = img_np[:, :, np.newaxis]

    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0)
    return img_tensor

# We don't need zlib and png decompression since those are lossless
# We have no need to compare them since we already know PSNR and SSIM will be infinity and 1.0