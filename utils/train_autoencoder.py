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
    cfg = config["train_autoencoder"]

    experiment.log_parameters(cfg)

    epochs = cfg["epochs"]
    lr = cfg["learning_rate"]
    weight_decay = cfg.get("weight_decay", 0)
    patience = cfg.get("patience", 5)
    min_delta = cfg.get("min_delta", 0.01)
    save_path = cfg.get("save_path", "autoencoder.pt")
    save_reconstructions = cfg.get("save_reconstructions", "reconstructions/")
    freeze_every_n = cfg.get("freeze_decoder_every_n", 3)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    best_score = -float("inf")
    best_model_wts = model.state_dict()
    counter = 0
    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Freeze decoder ogni n epoche
        if (epoch + 1) % freeze_every_n == 0:
            print(f"Epoch {epoch+1}: FREEZING decoder")
            for param in model.decoder_input.parameters():
                param.requires_grad = False
            for param in model.decoder_conv.parameters():
                param.requires_grad = False
        else:
            for param in model.decoder_input.parameters():
                param.requires_grad = True
            for param in model.decoder_conv.parameters():
                param.requires_grad = True

            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                weight_decay=weight_decay
            )

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            noisy_images = add_noise(images, noise_level=cfg.get("noise_level", 0.2))

            x_reconstructed, mu, logvar, _ = model(noisy_images, labels)

            loss, recon_loss, kl_loss = vae_loss(images, x_reconstructed, mu, logvar, beta=10)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item(), recon=recon_loss.item(), kl=kl_loss.item())

        avg_loss = running_loss / len(dataloader)
        avg_loss = torch.log(torch.tensor(avg_loss))
        train_losses.append(avg_loss)
        print(f"\nEpoch [{epoch + 1}/{epochs}] - Avg Loss: {avg_loss:.4f}")
        experiment.log_metric("train_loss", avg_loss, step=epoch + 1)

        latent_acc, latent_sil = evaluate_latent_space(model, dataloader, device)
        experiment.log_metric("latent_accuracy", latent_acc, step=epoch + 1)
        experiment.log_metric("latent_silhouette", latent_sil, step=epoch + 1)
        print(f"Latent Accuracy: {latent_acc:.4f} | Silhouette Score: {latent_sil:.4f}")

        composite_score = -avg_loss + 100 * latent_sil
        if composite_score - best_score > min_delta:
            print(f"Nuovo miglior modello trovato (loss: {composite_score:.4f} > {best_score:.4f})")
            best_score = composite_score
            best_model_wts = model.state_dict()
            counter = 0
        else:
            counter += 1

        model.eval()
        with torch.no_grad():
            sample_inputs = images[:8]
            sample_labels = labels[:8]

            noisy_inputs = add_noise(sample_inputs, noise_level=cfg.get("noise_level", 0.2))
            x_reconstructed, _, _, _ = model(noisy_inputs, sample_labels)

            comparison = torch.cat([sample_inputs, noisy_inputs, x_reconstructed], dim=0)
            image_path = os.path.join(save_reconstructions, f"epoch_{epoch + 1}_reconstruction_images.png")
            save_image(comparison, image_path, nrow=8)
            experiment.log_image(image_path, name=f"epoch_{epoch + 1}_reconstruction_images")

        if counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

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

    if best_model_wts:
        torch.save(best_model_wts, save_path)
        experiment.log_model("best_autoencoder_model", save_path)
        print(f"\nAddestramento completato. Miglior modello salvato in: {save_path} (loss: {best_score:.4f})")

def add_noise(images, noise_level=0.1):
    noise = torch.randn_like(images) * noise_level
    return torch.clamp(images + noise, -1.0, 1.0)
