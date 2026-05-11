import torch.nn as nn
import torch

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32, channels=1, height=28, width=28):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.channels = channels
        self.height = height
        self.width = width

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Linear(64, 256),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
    
    def decode(self, x):
        x = self.decoder(x)
        return x.view(x.size(0), self.channels, self.height, self.width)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        reconstruction = self.decode(z)

        return reconstruction, mu, logvar
    