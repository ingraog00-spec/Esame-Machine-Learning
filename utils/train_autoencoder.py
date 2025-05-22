import os
import torch
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.loss import vae_loss
from utils.latent_space_valuate import evaluate_latent_space
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
    best_sil_score = -1.0     # Per tenere traccia del miglior silhouette score ottenuto durante training

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

            # Aggiunta di noise alle immagini
            noisy_images = add_noise(images, noise_type=cfg.get("noise_type", "gaussian"), 
                             noise_level=cfg.get("noise_level", 0.1))

            # Forward pass con input rumoroso, ma target pulito
            x_reconstructed, mu, logvar, _ = model(noisy_images, labels)

            # Calcolo della loss
            loss, recon_loss, kl_loss = vae_loss(images, x_reconstructed, mu, logvar, beta=10)

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

        # --- Valutazione sullo spazio latente per early stopping guidato da metrica ---
        latent_acc, latent_sil = evaluate_latent_space(model, dataloader, device)
        experiment.log_metric("latent_accuracy", latent_acc, step=epoch + 1)
        experiment.log_metric("latent_silhouette", latent_sil, step=epoch + 1)
        print(f"Latent Accuracy: {latent_acc:.4f} | Silhouette Score: {latent_sil:.4f}")

        # --- Early Stopping e salvataggio modello migliore ---
        if latent_sil - best_sil_score > min_delta:
            best_sil_score = latent_sil
            best_loss = avg_loss
            best_model_wts = model.state_dict()
            counter = 0
        else:
            counter += 1

        # Salvataggio immagini di ricostruzione
        model.eval()
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

        # Attivazione early stopping se il modello non migliora per un numero di epoche consecutivo pari alla pazienza
        if counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    # Visualizzazione della curva della loss
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

    # Salvataggio del modello migliore
    if best_model_wts:
        torch.save(best_model_wts, save_path)
        experiment.log_model("best_autoencoder_model", save_path)
        print(f"\nAddestramento completato. Miglior modello salvato in: {save_path} (loss: {best_loss:.4f})")
    else:
        # Spera di non arrivare qua!
        print("\nNessun miglioramento durante l'addestramento. Modello non salvato.")


def add_noise(images, noise_type="gaussian", noise_level=0.1):
    if noise_type == "gaussian":
        noise = torch.randn_like(images) * noise_level
        return torch.clamp(images + noise, -1.0, 1.0)  # Mantieni nel range [-1, 1]
    elif noise_type == "salt_pepper":
        prob = noise_level
        rand = torch.rand_like(images)
        noisy = images.clone()
        noisy[rand < prob / 2] = -1.0     # pepper
        noisy[rand > 1 - prob / 2] = 1.0  # salt
        return noisy
    else:
        raise ValueError("Tipo di rumore non supportato")