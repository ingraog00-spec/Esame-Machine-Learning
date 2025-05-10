import torch
import torch.nn.functional as F

def vae_loss(x, x_reconstructed, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(x_reconstructed, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss
