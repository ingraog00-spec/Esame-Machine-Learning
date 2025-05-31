import torch.nn as nn
import torch

# BLOCCO RESIDUALE
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        # Layer interno
        self.layer = nn.Sequential(
            nn.Linear(dim, dim),         # Layer lineare
            nn.LayerNorm(dim),           # Normalizzazione
            nn.ReLU(),                   # Funzione di attivazione
            nn.Dropout(dropout)          # Dropout
        )

    def forward(self, x):
        # Connessione residua: somma input originale con l'output del layer interno
        return x + self.layer(x)

# CLASSIFICATORE PROFONDO CON RESIDUAL BLOCKS
class Classifier(nn.Module):
    def __init__(self, input_dim=256, num_classes=7):
        super().__init__()

        # STRATO DI INPUT
        self.input = nn.Sequential(
            nn.Linear(input_dim, 256),   # (B, 256) -> (B, 256)
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),         # (B, 512) -> (B, 256)
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # SEQUENZA RESIDUAL BLOCKS
        self.resblocks = nn.Sequential(
            ResidualBlock(128, dropout=0.3),
            ResidualBlock(128, dropout=0.3),
            ResidualBlock(128, dropout=0.3),
            ResidualBlock(128, dropout=0.2),
            ResidualBlock(128, dropout=0.2),
            ResidualBlock(128, dropout=0.2)
        )

        # STRATO DI USCITA
        self.output = nn.Linear(128, num_classes)  # (B, 256) -> (B, 7)

    def forward(self, x):
        x = self.input(x)
        x = self.resblocks(x)
        return self.output(x)

# LABEL SMOOTHING LOSS
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        """
        pred: output grezzi del classificatore (logits), dimensione (B, num_classes)
        target: etichette vere, dimensione (B,)
        """
        n_class = pred.size(1)

        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (n_class - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)  # Classe corretta = valore più alto

        # Calcola cross-entropy tra predizione e etichette
        return torch.mean(torch.sum(-true_dist * pred.log_softmax(dim=1), dim=1))
