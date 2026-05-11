import os
import torch

import matplotlib
matplotlib.use('Agg') # Switches to non-interactive backend, saves to files instead of opening windows
import matplotlib.pyplot as plt

from training.dataloader import DATASET_INFO


def save_loss_curve(train_losses, path):
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_reconstructions(model, test_loader, device, path, dataset, config):
    model_type = config["model"]
    model.eval()
    images, _ = next(iter(test_loader))
    images = images[:8].to(device)
    info = DATASET_INFO[dataset]

    with torch.no_grad():
        if model_type == "v3_vae":
            reconstructions, mu, logvar = model(images)

        elif model_type == "v1_linear_ae":
            reconstructions = model(images)

        elif model_type == "v2_convolutional_ae":
            reconstructions = model(images)
            
    if model_type == "v1_linear_ae":
        images = images.view(
            -1,
            info['channels'],
            info['size'],
            info['size']
        )

        reconstructions = reconstructions.view(
            -1,
            info['channels'],
            info['size'],
            info['size']
        )

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))

    for i in range(8):
        if info['channels'] == 1:
            axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
            axes[1, i].imshow(reconstructions[i].cpu().squeeze(), cmap='gray')
        else:
            axes[0, i].imshow(images[i].cpu().permute(1, 2, 0))
            axes[1, i].imshow(reconstructions[i].cpu().permute(1, 2, 0))

        axes[0, i].axis('off')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=8)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=8)
    plt.suptitle(f'{dataset} — reconstructions')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()