import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import io

def test_classifier(classifier_load, test_loader_cls, config, device, experiment):
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in test_loader_cls:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = classifier_load(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    experiment.log_metric("test_accuracy", accuracy)
    experiment.log_metric("test_precision", precision)
    experiment.log_metric("test_recall", recall)
    experiment.log_metric("test_f1_score", f1)

    print("---- Test Set Evaluation ----")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    report_str = classification_report(all_targets, all_preds, zero_division=0)
    print("\nClassification Report:\n", report_str)
    experiment.log_text(report_str)

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    experiment.log_image(buf, name="confusion_matrix_test.png")
    buf.close()
    plt.close()
