import torch.nn as nn

class LinearAutoencoder(nn.Module):
    model_type = "autoencoder"
    def __init__(self, input_dim=784, latent_dim=32, channels=1, height=28, width=28):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.channels = channels
        self.height = height
        self.width = width

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Linear(256, 64),
            nn.Linear(64, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Linear(64, 256),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)
    
    def decode(self, x):
        x = self.decoder(x)
        return x.view(x.size(0), self.channels, self.height, self.width)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z