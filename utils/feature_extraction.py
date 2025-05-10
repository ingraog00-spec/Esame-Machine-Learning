import torch
from tqdm import tqdm


def extract_embeddings(encoder, dataloader, device):
    encoder.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, lbls in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            mu, _ = encoder.encode(images)
            embeddings.append(mu.cpu())
            labels.extend(lbls.cpu().numpy())

    embeddings_tensor = torch.cat(embeddings)
    return embeddings_tensor, labels
