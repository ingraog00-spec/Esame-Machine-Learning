import torch
import torch.nn.functional as F

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