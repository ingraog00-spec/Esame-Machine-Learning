import torch
import torch.nn.functional as F
from torch import nn

# CENTER LOSS: penalizza la distanza tra le feature e i centroidi della propria classe
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        """
        Args:
            num_classes (int): numero totale di classi
            feat_dim (int): dimensione del vettore di embedding/feature
            device (torch.device): (CPU o GPU)
        """
        super(CenterLoss, self).__init__()

        # Inizializza i centroidi di ogni classe con valori casuali
        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim).to(device), requires_grad=True
        )

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

    def forward(self, features, labels):
        """
        Args:
            features (Tensor): vettori di feature estratti
            labels (Tensor): etichette intere

        Returns:
            loss (Tensor): media della distanza euclidea quadratica tra ogni feature
                           e il centro della propria classe
        """
        batch_size = features.size(0)  # numero di campioni nel batch

        # Seleziona il centro corrispondente a ciascun campione
        centers_batch = self.centers[labels]

        # Calcola la distanza euclidea quadratica tra ogni feature e il proprio centro
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
    recon_loss = F.mse_loss(x_reconstructed, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss