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

from training.dataloader import get_mnist_loaders

def load_model(config):
    model_name = config['model']

    if model_name == 'v1_linear_ae':
        from models.v1_linear_ae import LinearAutoencoder
        return LinearAutoencoder(latent_dim=config['latent_dim'])
    else:
        raise ValueError(f"Unknown model: {model_name}")

def save_reconstructions(model, test_loader, device, path):
    model.eval()
    images, _ = next(iter(test_loader))
    images = images[:8].to(device)

    with torch.no_grad():
        reconstructions, _ = model(images)

    fig, axes = plt.subplots(2, 8, figsize=(12, 3))

    for i in range(8):
        axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructions[i].cpu().squeeze(), cmap='gray')
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

    train_loader, test_loader = get_mnist_loaders(batch_size=config['batch_size'])
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

    torch.save(model.state_dict(), os.path.join(config['save_dir'], 'model.pt'))

    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(config['results_dir'], 'loss_curve.png'))
    plt.close()

    save_reconstructions(model, test_loader, device, os.path.join(config['results_dir'], 'reconstructions.png'))

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