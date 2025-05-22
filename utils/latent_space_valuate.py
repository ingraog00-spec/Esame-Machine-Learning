import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.svm import SVC
import torch
from sklearn.preprocessing import StandardScaler

def evaluate_latent_space(model, dataloader, device):
    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            lbls = lbls.to(device)
            _, mu, _, _ = model(images, lbls)  # Mu = media latente
            latents.append(mu.cpu().numpy())
            labels.extend(lbls.cpu().numpy())

    latents = np.concatenate(latents)
    labels = np.array(labels)

    # Normalizzazione per classificazione e silhouette
    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)

    # Classificatore ausiliario - SVM
    clf = SVC(kernel="linear", C=1.0, random_state=42)
    clf.fit(latents_scaled, labels)
    preds = clf.predict(latents_scaled)
    acc = accuracy_score(labels, preds)

    # Silhouette score
    sil_score = silhouette_score(latents_scaled, labels)

    return acc, sil_score