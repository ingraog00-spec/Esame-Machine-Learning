import torch
from sklearn.metrics import classification_report

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
    
    report = classification_report(all_labels, all_preds, output_dict=True)
    acc = report["accuracy"]
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]
    
    if experiment:
        experiment.log_metric(f"{log_prefix}accuracy", acc)
        experiment.log_metrics({
            f"{log_prefix}precision": precision,
            f"{log_prefix}recall": recall,
            f"{log_prefix}f1_score": f1
        })

    print(f"{title} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report
    }
