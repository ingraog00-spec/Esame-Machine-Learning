import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import comet_ml
from utils.loss import vae_loss


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

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    best_loss = float("inf")
    best_model_wts = None
    counter = 0
    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            x_reconstructed, mu, logvar, _ = model(images, labels)
            loss, recon_loss, kl_loss = vae_loss(images, x_reconstructed, mu, logvar, beta=0.1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), recon=recon_loss.item(), kl=kl_loss.item())

        avg_loss = running_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"\nEpoch [{epoch + 1}/{epochs}] - Avg Loss: {avg_loss:.4f}")
        experiment.log_metric("train_loss", avg_loss, step=epoch + 1)

        # Early stopping
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            best_model_wts = model.state_dict()
            counter = 0
        else:
            counter += 1
            print(f"No improvement for {counter} epochs")

        # Save reconstructions
        model.eval()
        with torch.no_grad():
            sample_inputs = images[:8]
            sample_labels = labels[:8]
            x_reconstructed, _, _, _ = model(sample_inputs, sample_labels)
            comparison = torch.cat([sample_inputs, x_reconstructed])
            image_path = os.path.join(save_reconstructions, f"epoch_{epoch + 1}_recon.png")
            os.makedirs(save_reconstructions, exist_ok=True)
            save_image(comparison, image_path, nrow=8)
            experiment.log_image(image_path, name=f"epoch_{epoch + 1}_reconstruction")

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
        os.makedirs("./images", exist_ok=True)
        plt.savefig(loss_curve_path)
        experiment.log_image(loss_curve_path)

    # Save best model
    if best_model_wts:
        torch.save(best_model_wts, save_path)
        experiment.log_model("best_autoencoder_model", save_path)
        print(f"\nTraining complete. Best model saved at: {save_path} (loss: {best_loss:.4f})")
    else:
        print("\nNo improvement during training. Model not saved.")
