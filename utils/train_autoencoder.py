import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
import torch
torch.set_num_threads(4)
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
from math import log
import matplotlib.pyplot as plt
from utils.loss import vae_loss, CenterLoss
from utils.latent_space_valuate import evaluate_latent_space
import comet_ml

def train_autoencoder(model, dataloader, config, device, experiment):
    cfg = config["train_autoencoder"]
    experiment.log_parameters(cfg)

    # Estrae gli iperparametri dal file di configurazione
    epochs = cfg["epochs"]
    lr = cfg["learning_rate"]
    weight_decay = cfg.get("weight_decay", 0)
    patience = cfg.get("patience", 5)
    min_delta = cfg.get("min_delta", 0.01)
    save_path = cfg.get("save_path", "autoencoder.pt")
    save_reconstructions = cfg.get("save_reconstructions", "reconstructions/")
    beta_max = cfg.get("beta_max", 10)
    beta_midpoint = cfg.get("beta_midpoint", 10)
    beta_steepness = cfg.get("beta_steepness", 0.3)
    sil_weight = cfg.get("silhouette_weight", 10)
    lambda_center = cfg.get("center_loss_weight", 1.0)

    sparsity_weight = cfg.get("sparsity_weight", 0.1)  # nuovo iperparametro
    rho_sparsity = cfg.get("sparsity_target", 0.05)    # attivazione desiderata

    center_loss_fn = CenterLoss(num_classes=model.num_classes, feat_dim=model.latent_dim, device=device)

    # Ottimizzatore Adam combinazione di pesi e centroidi
    optimizer = optim.Adam(
        list(model.parameters()) + list(center_loss_fn.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    model.to(device)

    best_score = -float("inf")
    best_model_wts = model.state_dict()
    counter = 0

    # Per logging e controllo delle condizioni di addestramento
    train_losses = []
    kl_losses_per_epoch = []

    for epoch in range(epochs):
        model.train()
        kl_epoch = []
        running_loss = 0.0

        # beta-annealing
        beta = sigmoid_annealing(epoch, beta_max, midpoint=beta_midpoint, steepness=beta_steepness)

        # Barra di avanzamento per ogni batch
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            # Applica rumore alle immagini di input
            noisy_images = add_noise(images, noise_level=cfg.get("noise_level", 0.2))

            # Forward pass del modello
            x_reconstructed, mu, logvar, z = model(noisy_images, labels)

            # Calcola la loss VAE
            loss_vae, recon_loss, kl_loss = vae_loss(images, x_reconstructed, mu, logvar, beta=beta)
            loss_center = center_loss_fn(z, labels)
            loss_sparsity = sparsity_loss(z, rho=rho_sparsity)
            total_loss = loss_vae + lambda_center * loss_center + sparsity_weight * loss_sparsity

            # Backpropagation e aggiornamento pesi
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            kl_epoch.append(kl_loss.item())
            running_loss += total_loss.item()

            # Aggiorna la barra
            progress_bar.set_postfix(loss=total_loss.item(), recon=recon_loss.item(), kl=kl_loss.item())

        # Calcolo della loss media per epoca
        avg_loss = running_loss / len(dataloader)
        avg_loss = log(avg_loss)
        train_losses.append(avg_loss)

        avg_kl_loss = sum(kl_epoch) / len(kl_epoch)
        kl_losses_per_epoch.append(avg_kl_loss)
        experiment.log_metric("avg_kl_loss", avg_kl_loss, step=epoch + 1)

        print(f"\nEpoch [{epoch + 1}/{epochs}] - Avg Loss: {avg_loss:.4f} | β: {beta:.2f}")
        experiment.log_metric("train_loss", avg_loss, step=epoch + 1)
        experiment.log_metric("kl_beta", beta, step=epoch + 1)
        experiment.log_metric("center_loss", loss_center.item(), step=epoch + 1)
        experiment.log_metric("sparsity_loss", loss_sparsity.item(), step=epoch + 1)

        # Valuta la qualità dello spazio latente usando la silhouette
        latent_sil = evaluate_latent_space(model, dataloader, device, experiment, epoch + 1, "./images/latent_space")
        experiment.log_metric("latent_silhouette", latent_sil, step=epoch + 1)

        # Calcolo score combinato per l'early stopping
        composite_score = -avg_loss + sil_weight * latent_sil
        if composite_score - best_score > min_delta:
            print(f"Nuovo miglior modello trovato (score: {composite_score:.4f} > {best_score:.4f})")
            best_score = composite_score
            best_model_wts = model.state_dict()
            counter = 0
        else:
            counter += 1

        # Salva ricostruzioni tra: input, rumore e output
        model.eval()
        with torch.no_grad():
            class_examples = []
            class_labels_found = set()

            # Trova una immagine per ogni classe
            for img, lbl in zip(images, labels):
                if lbl.item() not in class_labels_found:
                    class_examples.append((img, lbl))
                    class_labels_found.add(lbl.item())
                if len(class_examples) == model.num_classes:
                    break

            if len(class_examples) < model.num_classes:
                print(f"Attenzione: trovate solo {len(class_examples)} classi nel batch attuale.")

            inputs, lbls = zip(*class_examples)
            inputs = torch.stack(inputs).to(device)
            lbls = torch.tensor(lbls).to(device)

            noisy_inputs = add_noise(inputs, noise_level=cfg.get("noise_level", 0.2))
            x_reconstructed, _, _, _ = model(noisy_inputs, lbls)

            # Concatena: originali, noisy, ricostruiti
            comparison = torch.cat([inputs, noisy_inputs, x_reconstructed], dim=0)

            image_path = os.path.join(save_reconstructions, f"epoch_{epoch + 1}_reconstruction_images.png")
            save_image(comparison, image_path, nrow=model.num_classes)
            experiment.log_image(image_path, name=f"epoch_{epoch + 1}_reconstruction_images")

        # Interrompe l'addestramento se non ci sono miglioramenti per 'patience' epoche
        if counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    # Grafico della curva di loss
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

    # Miglior modello ottenuto
    if best_model_wts:
        torch.save(best_model_wts, save_path)
        experiment.log_model("best_autoencoder_model", save_path)
        print(f"\nAddestramento completato. Miglior modello salvato in: {save_path} (loss: {best_score:.4f})")

    # Grafico della perdita KL per epoca
    if kl_losses_per_epoch:
        plt.figure()
        plt.plot(range(1, len(kl_losses_per_epoch) + 1), kl_losses_per_epoch, marker='x', color='orange')
        plt.title("KL Divergence per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Average KL Divergence")
        plt.grid(True)
        plt.tight_layout()
        kl_curve_path = os.path.join("./images", "kl_loss_curve.png")
        plt.savefig(kl_curve_path)
        experiment.log_image(kl_curve_path)
        plt.close()

# Rumore gaussiano
def add_noise(images, noise_level=0.1):
    noise = torch.randn_like(images) * noise_level
    return torch.clamp(images + noise, -1.0, 1.0)

def sigmoid_annealing(epoch, max_beta, midpoint=10, steepness=1.0):
    x = torch.tensor(-steepness * (epoch - midpoint), dtype=torch.float32)
    return float(max_beta / (1 + torch.exp(x)))

def sparsity_loss(z, rho=0.05):
    """
    Penalizza le attivazioni latenti troppo elevate rispetto alla media desiderata rho.
    Usa la KL divergence tra rho e la media delle attivazioni sigmoidee di z.
    """
    rho_hat = torch.mean(torch.sigmoid(z), dim=0)
    rho_tensor = torch.full_like(rho_hat, rho)
    kl_div = rho_tensor * torch.log(rho_tensor / (rho_hat + 1e-8)) + \
             (1 - rho_tensor) * torch.log((1 - rho_tensor) / (1 - rho_hat + 1e-8))
    return kl_div.sum()
