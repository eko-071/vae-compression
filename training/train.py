import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import torch.nn as nn

from training.dataloader import get_dataloaders, get_input_dim, DATASET_INFO
from training.visualize import save_loss_curve, save_reconstructions
from training.model_loader import load_model

def vae_loss(reconstructions, images, mu, logvar, beta=1.0):
    recon_loss = nn.functional.binary_cross_entropy(reconstructions,images,reduction='sum') / images.size(0)
    # kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1).mean()
    kl_loss = kl_loss / logvar.size(1)
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)

    train_loader, test_loader = get_dataloaders(
        dataset=config['dataset'],
        batch_size=config['batch_size']
    )

    model = load_model(config).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    beta = config.get('beta', 1)

    train_losses = []

    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0

        for images, _ in train_loader:
            images = images.to(device)

            if model.model_type == 'vae':
                reconstructions, mu, logvar = model(images)
                loss, recon_loss, kl_loss = vae_loss(
                    reconstructions, images, mu, logvar, beta
                )
            else:
                reconstructions, _ = model(images)
                loss = nn.functional.mse_loss(reconstructions, images)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch}/{config['epochs']}: Loss {avg_loss:.6f} | {elapsed:.1f}s")

    filename = f"{config['dataset']}_ld{config['latent_dim']}.pt"
    torch.save(model.state_dict(), os.path.join(config['save_dir'], filename))

    filename = f"{config['dataset']}_ld{config['latent_dim']}_loss_curve.png"
    save_loss_curve(train_losses, os.path.join(config['results_dir'], filename))

    filename = f"{config['dataset']}_ld{config['latent_dim']}_reconstructions.png"
    save_reconstructions(
        model, test_loader, device,
        os.path.join(config['results_dir'], filename),
        config['dataset'], config
    )

    print("Training complete.")
    print("Model saved to", config['save_dir'])
    print("Results saved to", config['results_dir'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train(config)