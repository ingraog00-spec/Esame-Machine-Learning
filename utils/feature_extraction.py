import torch
from tqdm import tqdm

def extract_embeddings(model, dataloader, device, mode="mu"):
    """
    Estrae rappresentazioni latenti dallo spazio latente di un VAE condizionale.
    
    Args:
        model: Modello VAE
        dataloader: Dataloader PyTorch
        device: torch.device
        mode: 'mu' | 'z' — indica quale embedding estrarre
    
    Returns:
        embeddings: array numpy (n_samples, latent_dim)
        labels: array numpy (n_samples,)
    """
    assert mode in ["mu", "z"], "mode deve essere 'mu' o 'z'"
    
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for images, lbls in tqdm(dataloader, desc="Estrazione embeddings"):
            images = images.to(device)
            lbls = lbls.to(device)
            
            if mode == "mu": # mode == "mu", estrae l'embedding mu
                mu, _ = model.encode(images, lbls)
                emb = mu
            else:  # mode == "z", estrae l'embedding z cioè lo spazio latente
                _, _, _, z = model(images, lbls)
                emb = z
            
            embeddings.append(emb.cpu())
            labels.append(lbls.cpu())
    
    embeddings_tensor = torch.cat(embeddings, dim=0).numpy()
    labels_array = torch.cat(labels, dim=0).numpy()
    
    return embeddings_tensor, labels_array
