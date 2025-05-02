import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_classifier(model, train_loader, val_loader, config, device):
    cfg = config["train_classifier"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0))
    model.to(device)

    best_acc = 0
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

        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)
        acc = correct / total
        print(f"Epoch [{epoch+1}/{cfg['epochs']}] - Loss: {avg_loss:.4f} - Val Acc: {acc:.4f}")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), cfg["save_path"])
            print(f"New best model saved with acc: {acc:.4f}")
