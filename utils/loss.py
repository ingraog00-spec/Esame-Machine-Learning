import torch
import torch.nn.functional as F
from torch import nn

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device), requires_grad=False)
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers[labels]
        loss = (features - centers_batch).pow(2).sum() / batch_size
        return loss

def vae_loss(x, x_reconstructed, mu, logvar, beta=1.0):
    """
    Calcola la loss per un VAE: somma tra errore di ricostruzione (MSE) e divergenza KL, con peso beta.

    Args:
        x: input originale.
        x_reconstructed: output ricostruito dal decoder.
        mu: media della distribuzione latente.
        logvar: log-varianza della distribuzione latente.
        beta: peso per la KL (default 1.0).

    Returns:
        total_loss, recon_loss, kl_loss
    """
    recon_loss = F.mse_loss(x_reconstructed, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss