import comet_ml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

def train_classifier(model, train_loader, val_loader, config, device, experiment):
    cfg = config["train_classifier"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0))
    model.to(device)

    best_acc = 0
    train_losses = []

    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}", leave=False)
        
        for x_batch, y_batch in progress_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        experiment.log_metric("train_loss", avg_loss, step=epoch + 1)

        # Eval
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_val.cpu().numpy())

        acc = correct / total
        experiment.log_metric("val_accuracy", acc, step=epoch + 1)
        print(f"Epoch [{epoch+1}/{cfg['epochs']}] - Loss: {avg_loss:.4f} - Val Acc: {acc:.4f}")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            save_path = cfg["save_path"]
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with acc: {acc:.4f}")

    report = classification_report(all_labels, all_preds, output_dict=True)
    experiment.log_metrics({
        "final_val_accuracy": report["accuracy"],
        "final_val_precision": report["weighted avg"]["precision"],
        "final_val_recall": report["weighted avg"]["recall"],
        "final_val_f1_score": report["weighted avg"]["f1-score"]
    })

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    confusion_matrix_path = "./reconstructions/confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    experiment.log_image(confusion_matrix_path)

    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    loss_curve_path = "./reconstructions/train_loss_curve.png"
    plt.savefig(loss_curve_path)
    experiment.log_image(loss_curve_path)
