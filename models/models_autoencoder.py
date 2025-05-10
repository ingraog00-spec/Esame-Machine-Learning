import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvVariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: Conv layers + Flatten
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 128, 128]
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 64, 64]
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 32, 32]
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 16, 16]
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Flatten()  # [B, 256*16*16]
        )

        self.flatten_dim = 256 * 16 * 16

        # Latent space
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder: Dense → ConvTranspose layers
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (256, 16, 16)),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 32, 32]
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 64, 64]
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 128, 128]
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # [B, 3, 256, 256]
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        x_reconstructed = self.decoder_conv(h)
        return x_reconstructed

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar, z
