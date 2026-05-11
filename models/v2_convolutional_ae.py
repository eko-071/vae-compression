import torch
import torch.nn as nn


class ConvolutionalAutoencoder(nn.Module):
    model_type = "autoencoder"
    def __init__(self, latent_dim=32, channels=1, height=28, width=28):
        super().__init__()

        self.latent_dim = latent_dim
        self.channels = channels
        self.height = height
        self.width = width

        # Encoder
        self.encoder = nn.Sequential(
            # [B, 1, 28, 28]
            nn.Conv2d(
                in_channels=channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            # [B, 32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            # [B, 64, 7, 7]
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            x = self.encoder(dummy)
            self.flatten_dim = x.view(1, -1).shape[1]

        # Latent projection
        self.fc_encoder = nn.Linear(
            self.flatten_dim,
            latent_dim
        )

        # Decoder projection
        self.fc_decoder = nn.Linear(
            latent_dim,
            self.flatten_dim
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            # [B, 32, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            # [B, 1, 28, 28]
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.fc_encoder(x)

        return z

    def decode(self, z):
        x = self.fc_decoder(z)
        conv_h = self.height // 4
        conv_w = self.width // 4
        x = x.view(z.size(0), 64, conv_h, conv_w)
        x = self.decoder(x)

        return x

    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        
        return reconstruction, z