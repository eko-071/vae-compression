import torch.nn as nn

class LinearAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.Linear(256, 64),
            nn.Linear(64, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Linear(64, 256),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)
    
    def decode(self, x):
        x = self.decoder(x)
        return x.view(x.size(0), 1, 28, 28)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z