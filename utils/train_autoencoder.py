import os
import torch
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.loss import vae_loss
import numpy as np
from sklearn.manifold import TSNE
import comet_ml

def train_autoencoder(model, dataloader, config, device, experiment):
    # Configurazione
    cfg = config["train_autoencoder"]

    # Log dei parametri di training su Comet.ml
    experiment.log_parameters(cfg)

    epochs = cfg["epochs"]                                                     # Numero di epoche di addestramento
    lr = cfg["learning_rate"]                                                  # Learning rate
    weight_decay = cfg.get("weight_decay", 0)                                  # Termine di regolarizzazione
    patience = cfg.get("patience", 5)                                          # Numero di epoche senza miglioramento per attivare early stopping
    min_delta = cfg.get("min_delta", 0.01)                                     # Soglia minima di miglioramento per considerare progresso reale
    save_path = cfg.get("save_path", "autoencoder.pt")                         # Percorso per salvataggio modello migliore
    save_reconstructions = cfg.get("save_reconstructions", "reconstructions/") # Cartella salvataggio immagini ricostruite

    # Inizializzazione ottimizzatore Adam
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)  # Spostamento modello sul device (CPU-GPU-MPS)

    best_loss = float("inf")  # Per tenere traccia della migliore loss ottenuta durante training
    best_model_wts = None     # Per salvare i pesi del modello migliore
    counter = 0               # Contatore per early stopping
    train_losses = []         # Lista per registrare la loss media per ogni epoca

    # Loop principale di training
    for epoch in range(epochs):
        model.train()         # Modalità training attiva
        running_loss = 0.0    # Accumulatore della loss

        # progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        # Iterazione sui batch del DataLoader
        for images, labels in progress_bar:
            # Spostamento batch sul device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            x_reconstructed, mu, logvar, _ = model(images, labels)

            # Calcolo della loss
            loss, recon_loss, kl_loss = vae_loss(images, x_reconstructed, mu, logvar, beta=2)

            # Azzeramento dei gradienti
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Aggiornamento dei pesi
            optimizer.step()

            running_loss += loss.item()

            # Aggiornamento della progress bar
            progress_bar.set_postfix(loss=loss.item(), recon=recon_loss.item(), kl=kl_loss.item())

        # Calcolo della loss media per l'epoca corrente
        avg_loss = running_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"\nEpoch [{epoch + 1}/{epochs}] - Avg Loss: {avg_loss:.4f}")

        experiment.log_metric("train_loss", avg_loss, step=epoch + 1)

        # --- Early Stopping ---
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            best_model_wts = model.state_dict()  # Salvataggio pesi del modello migliore finora
            counter = 0                          # Reset contatore di epoche senza miglioramento
        else:
            counter += 1                         # Incremento contatore se nessun miglioramento
            print(f"Nessun miglioramento per {counter} epoche")

        # Salvataggio immagini di ricostruzione
        model.eval()  # Modalità evaluation: disabilita dropout e batchnorm
        with torch.no_grad():
            sample_inputs = images[:8]
            sample_labels = labels[:8]
            x_reconstructed, _, _, _ = model(sample_inputs, sample_labels)

            # Concateno le immagini
            comparison = torch.cat([sample_inputs, x_reconstructed])
            image_path = os.path.join(save_reconstructions, f"epoch_{epoch + 1}_recon.png")
            save_image(comparison, image_path, nrow=8) # Salvo immagine
            # Log immagine su Comet
            experiment.log_image(image_path, name=f"epoch_{epoch + 1}_reconstruction")

        log_latent_space(model, dataloader, device, labels, experiment, epoch, save_path="./images/latent_spaces/")

        # Attivazione early stopping se il modello non migliora per un numero di epoche consecutivo pari alla pazienza
        if counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    # Visualizzazione e salvataggio del grafico della curva di loss durante il training
    if train_losses:
        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
        plt.title("Training Loss Curve (Autoencoder)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        loss_curve_path = os.path.join("./images", "autoencoder_loss_curve.png")
        plt.savefig(loss_curve_path)
        experiment.log_image(loss_curve_path)
        plt.close()

    # Salvataggio finale dei pesi del modello corrispondenti alla migliore performance osservata
    if best_model_wts:
        torch.save(best_model_wts, save_path)
        experiment.log_model("best_autoencoder_model", save_path)
        print(f"\nAddestramento completato. Miglior modello salvato in: {save_path} (loss: {best_loss:.4f})")
    else:
        # Spera di non arrivare qua!
        print("\nNessun miglioramento durante l'addestramento. Modello non salvato.")


def log_latent_space(model, dataloader, device, inv_label_map, experiment, epoch, save_path="./images/latent_spaces"):
    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            lbls = lbls.to(device)
            _, mu, _, _ = model(images, lbls)
            latents.append(mu.cpu().numpy())
            labels.extend(lbls.cpu().numpy())

    latents = np.concatenate(latents, axis=0)
    labels = np.array(labels)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    latents_2d = tsne.fit_transform(latents)

    # Etichette leggibili
    label_names = [inv_label_map[int(l)] if int(l) in inv_label_map else str(int(l)) for l in labels]

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.title(f"Latent Space - Epoch {epoch + 1}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.colorbar(scatter, ticks=range(len(inv_label_map)), label="Classe")

    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, f"latent_epoch_{epoch + 1}.png")
    plt.tight_layout()
    plt.savefig(path)
    experiment.log_image(path, name=f"Latent Space Epoch {epoch + 1}")
    plt.close()