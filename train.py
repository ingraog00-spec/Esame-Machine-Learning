from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision.utils import save_image

def train_autoencoder(model, dataloader, config, device, experiment):
    cfg = config["train_autoencoder"]

    experiment.log_parameters(cfg)

    epochs = cfg["epochs"]
    lr = cfg["learning_rate"]
    weight_decay = cfg.get("weight_decay", 0)
    patience = cfg.get("patience", 5)
    min_delta = cfg.get("min_delta", 0.01)
    save_path = cfg.get("save_path", "autoencoder.pth")
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    best_loss = float("inf")
    best_model_wts = None
    counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for images, _ in progress_bar:
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(dataloader)
        print(f"\nEpoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.4f}")
        experiment.log_metric("train_loss", avg_loss, step=epoch + 1)

        # Early stopping
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            best_model_wts = model.state_dict()
            counter = 0
        else:
            counter += 1
            print(f"No improvement for {counter} epochs")

        # Salva immagini
        model.eval()
        with torch.no_grad():
            sample_inputs = images[:8]
            sample_outputs = model(sample_inputs)
            comparison = torch.cat([sample_inputs, sample_outputs])
            image_path = f"{save_dir}/epoch_{epoch+1}_recon.png"
            save_image(comparison, image_path, nrow=8)
            experiment.log_image(image_path, name=f"epoch_{epoch+1}_reconstruction")

        if counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    if best_model_wts:
        torch.save(best_model_wts, save_path)
        experiment.log_model("best_model", save_path)
        print(f"\nTraining complete. Best model saved at: {save_path} (loss: {best_loss:.4f})")
    else:
        print("\nNo improvement during training. Model not saved.")
