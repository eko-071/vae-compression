import io
import zlib
import torch
import numpy as np
from PIL import Image

def compress_with_model(model, image_tensor):
    """
    Compresses a single image and returns latent bytes and size in bytes.
    image_tensor: torch tensor of shape (1, C, H, W) with values in range [0, 1]
    """
    model.eval()
    with torch.no_grad():
        if hasattr(model, "encode_deterministic"):
            z = model.encode_deterministic(image_tensor)
        else:
            z = model.encode(image_tensor)
    
    latent_np = z.cpu().numpy().astype(np.float32)
    latent_bytes = latent_np.tobytes()
    
    return latent_bytes, latent_np.nbytes

def compress_with_jpeg(image_tensor, quality=75):
    """
    Compresses a single image with JPEG and returns jpeg bytes and size in bytes.
    quality: 1-95, lower is more compressed. 
    """
    img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    if img_np.shape[2] == 1:
        img_pil = Image.fromarray(img_np.squeeze(-1), mode='L')
    else:
        img_pil = Image.fromarray(img_np, mode='RGB')
    
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality)
    jpeg_bytes = buffer.getvalue()
    return jpeg_bytes, len(jpeg_bytes)

def compress_with_png(image_tensor):
    """
    Compresses a single image with PNG (lossless) and returns png bytes and size in bytes.
    """
    img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    if img_np.shape[2] == 1:
        img_pil = Image.fromarray(img_np.squeeze(-1), mode='L')
    else:
        img_pil = Image.fromarray(img_np, mode='RGB')
    
    buffer = io.BytesIO()
    img_pil.save(buffer, format='PNG')
    png_bytes = buffer.getvalue()
    return png_bytes, len(png_bytes)

def compress_with_zlib(image_tensor):
    """
    Compresses a single image with zlib, a naive general purpose compression.
    Returns compressed bytes and size in bytes.    
    """
    img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    raw_bytes = img_np.tobytes()
    compressed_bytes = zlib.compress(raw_bytes, level=9)
    return compressed_bytes, len(compressed_bytes)

def original_size(image_tensor):
    """
    Returns the size of the image in bytes as uint8 (1 byte per pixel value).
    Used as baseline for comparison.
    """
    return image_tensor.numel()