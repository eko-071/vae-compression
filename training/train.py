import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg') # Switches to non-interactive backend, saves to files instead of opening windows
import matplotlib.pyplot as plt

from training.dataloader import get_dataloaders, get_input_dim, DATASET_INFO

def load_model(config):
    dataset = config['dataset']
    info = DATASET_INFO[dataset]
    input_dim = get_input_dim(dataset)
    model_name = config['model']

    if model_name == 'v1_linear_ae':
        from models.v1_linear_ae import LinearAutoencoder
        return LinearAutoencoder(
            input_dim=input_dim,
            latent_dim=config['latent_dim'],
            channels=info['channels'],
            height=info['height'],
            width=info['width']
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def save_reconstructions(model, test_loader, device, path, dataset):
    model.eval()
    images, _ = next(iter(test_loader))
    images = images[:8].to(device)

    with torch.no_grad():
        reconstructions, _ = model(images)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))

    for i in range(8):
        if dataset == 'mnist':
            axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
            axes[1, i].imshow(reconstructions[i].cpu().squeeze(), cmap='gray')

        elif dataset == 'cifar10':
            axes[0, i].imshow(images[i].cpu().permute(1, 2, 0))
            axes[1, i].imshow(reconstructions[i].cpu().permute(1, 2, 0))

        else:
            raise ValueError(f"Unsupported dataset for visualization: {dataset}")

        axes[0, i].axis('off')
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Original', fontsize=8)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=8)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)

    train_loader, test_loader = get_dataloaders(dataset=config['dataset'], batch_size=config['batch_size'])
    model = load_model(config).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    train_losses = []

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        epoch_loss = 0

        for images, _ in train_loader:
            images = images.to(device)

            reconstructions, _ = model(images)
            loss = nn.functional.mse_loss(reconstructions, images)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch}/{config['epochs']}: Loss is {avg_loss:.6f}")

    filename = f"{config['dataset']}.pt"
    torch.save(model.state_dict(), os.path.join(config['save_dir'], filename))

    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss')
    filename = f"{config['dataset']}_loss_curve.png"
    plt.savefig(os.path.join(config['results_dir'], filename))
    plt.close()

    filename = f"{config['dataset']}_reconstructions.png"
    save_reconstructions(model, test_loader, device, os.path.join(config['results_dir'], filename), config['dataset'])

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