import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def test_classifier(model, data_loader, device, experiment=None, title="Model Evaluation", log_prefix=""):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = correct / total
    print(f"\n{title} Accuracy: {acc:.4f}")
    report = classification_report(all_labels, all_preds, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    if experiment:
        experiment.log_metric(f"{log_prefix}accuracy", acc)
        experiment.log_metrics({
            f"{log_prefix}precision": report["weighted avg"]["precision"],
            f"{log_prefix}recall": report["weighted avg"]["recall"],
            f"{log_prefix}f1_score": report["weighted avg"]["f1-score"]
        })

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{title} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        cm_path = f"./reconstructions/{log_prefix}confusion_matrix.png"
        os.makedirs("./reconstructions", exist_ok=True)
        plt.savefig(cm_path)
        experiment.log_image(cm_path)
        plt.close()

    return acc, report
