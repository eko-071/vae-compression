import torch
import torch.nn as nn


class ConvolutionalVAE(nn.Module):
    model_type = 'vae'

    def __init__(self, latent_dim=32, channels=1, height=28, width=28):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.height = height
        self.width = width

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            x = self.encoder(dummy)
            self.flatten_dim = x.view(1, -1).shape[1]
            self.conv_h = x.shape[2]
            self.conv_w = x.shape[3]

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def encode_deterministic(self, x):
        mu, _ = self.encode(x)
        return mu

    def decode(self, z):
        x = self.fc_decoder(z)
        x = x.view(z.size(0), 64, self.conv_h, self.conv_w)
        return self.decoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar