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


def load_model(config):
    from models.v1_linear_ae import LinearAutoencoder
    from models.v3_vae import VariationalAutoencoder
    from models.v2_convolutional_ae import ConvolutionalAutoencoder

    dataset = config['dataset']
    info = DATASET_INFO[dataset]
    input_dim = get_input_dim(dataset)
    model_name = config['model']

    if model_name == 'v1_linear_ae':
        return LinearAutoencoder(
            input_dim=input_dim,
            latent_dim=config['latent_dim'],
            channels=info['channels'],
            height=info['height'],
            width=info['width'],
        )
    
    elif model_name == 'v3_vae':
        return VariationalAutoencoder(
            input_dim=input_dim,
            latent_dim=config['latent_dim'],
            channels=info['channels'],
            height=info['height'],
            width=info['width'],
        )
    elif model_name == 'v2_convolutional_ae':
        return ConvolutionalAutoencoder(
            latent_dim=config['latent_dim'],
            channels=info['channels'],
            height=info['height'],
            width=info['width'],
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

def vae_loss(reconstructions, images, mu, logvar):

    recon_loss = nn.functional.mse_loss(
        reconstructions,
        images,
        reduction='mean'
    )

    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    beta = 0.01
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
    
    # return recon_loss + 0.001*kl_loss

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

    train_losses = []

    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0

        for images, _ in train_loader:
            images = images.to(device)

            if config['model'] == 'v3_vae':

                reconstructions, mu, logvar = model(images)

                loss, recon_loss, kl_loss = vae_loss(
                    reconstructions,
                    images,
                    mu,
                    logvar
                )

            elif config['model'] == 'v1_linear_ae':

                reconstructions = model(images)

                loss = nn.functional.mse_loss(
                    reconstructions,
                    images
                )
            
            elif config['model'] == 'v2_convolutional_ae':

                reconstructions = model(images)

                loss = nn.functional.mse_loss(
                    reconstructions,
                    images
                )

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

    # with open(args.config, 'rb') as f:
    #     raw = f.read()
    #     print(raw[:50])

    # print("CONFIG LOADED:", config)
    # print("CONFIG PATH:", args.config)
    # print("ABS PATH:", os.path.abspath(args.config))
    # print("FILE EXISTS:", os.path.exists(args.config))

    train(config)