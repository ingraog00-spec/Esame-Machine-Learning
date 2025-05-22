import torch
from tqdm import tqdm

def extract_embeddings(encoder, dataloader, device):
    # Imposta l'encoder in modalità valutazione (eval)
    encoder.eval()
    embeddings = []
    labels = []

    # Disabilita il calcolo dei gradienti
    with torch.no_grad():
        # Itera sul dataloader mostrando una barra di avanzamento
        for images, lbls in tqdm(dataloader, desc="Estrazione embeddings"):
            # Sposta immagini e label sul device specificato (CPU/GPU)
            images = images.to(device)
            lbls_tensor = lbls.to(device)
            # Passa immagini e label attraverso l'encoder per ottenere gli embeddings (mu)
            mu, _ = encoder.encode(images, lbls_tensor)
            # Salva gli embeddings sulla CPU
            embeddings.append(mu.cpu())
            # Salva le label come array numpy
            labels.extend(lbls.cpu().numpy())

    # Concatena tutti gli embeddings in un unico tensore
    embeddings_tensor = torch.cat(embeddings)
    return embeddings_tensor, labels

def extract_embeddings_latent_space(model, dataloader, device):
    model.eval()
    zs = []
    ys = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Estrazione embeddings latent space"):
            images = images.to(device)
            labels = labels.to(device)
            _, _, _, z = model(images, labels)
            zs.append(z.cpu())
            ys.append(labels.cpu())
    zs = torch.cat(zs, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    return zs, ys

