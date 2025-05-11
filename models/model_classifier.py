import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.layer(x)

class Classifier(nn.Module):
    def __init__(self, input_dim=256, num_classes=7):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, 512),       # Espansione
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),             # Compressione
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.resblocks = nn.Sequential(
            ResidualBlock(256, dropout=0.3),
            ResidualBlock(256, dropout=0.3),
            ResidualBlock(256, dropout=0.3),
            ResidualBlock(256, dropout=0.2),
            ResidualBlock(256, dropout=0.2),
            ResidualBlock(256, dropout=0.2)
        )

        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.input(x)
        x = self.resblocks(x)
        return self.output(x)

# Ensemble di più classificatori
class EnsembleClassifier(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        logits = [model(x) for model in self.models]
        return torch.stack(logits).mean(dim=0)

# Loss con Label Smoothing
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_class = pred.size(1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (n_class - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred.log_softmax(dim=1), dim=1))
