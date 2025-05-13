import comet_ml
import torch
from torch.utils.data import TensorDataset, DataLoader
from comet_ml import Experiment
import yaml
from models.model_classifier import Classifier, EnsembleClassifier
from utils.train_classifier import train_classifier
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

with open("config.yml") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = torch.load("./save_model_embeddings/embeddings.pt", weights_only=False)

train_embeddings, train_labels = data["train"]
val_embeddings, val_labels = data["val"]
test_embeddings, test_labels = data["test"]

train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
test_labels = torch.tensor(test_labels)

embeddings = torch.cat([train_embeddings, val_embeddings, test_embeddings], dim=0)
labels = torch.cat([train_labels, val_labels, test_labels], dim=0)

""" print(f"Embeddings shape: {embeddings.shape}")
print(f"Labels shape: {labels.shape}") """

train_dataset = TensorDataset(train_embeddings, train_labels)
val_dataset = TensorDataset(val_embeddings, val_labels)
test_dataset = TensorDataset(test_embeddings, test_labels)

train_loader_cls = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader_cls = DataLoader(val_dataset, batch_size=32)
test_loader_cls = DataLoader(test_dataset, batch_size=32)

experiment = Experiment()
experiment.set_name("test del classificatore")

n_models = 3
ensemble_models = []

val_accuracies = []
val_precisions = []
val_recalls = []
val_f1s = []

for i in range(n_models):
    print(f"\n--- Training model {i+1}/{n_models} ---")
    set_seed(42 + i)
    model = Classifier(input_dim=embeddings.shape[1], num_classes=7)

    metrics = train_classifier(model, train_loader_cls, val_loader_cls, config, device, experiment=experiment)

    val_accuracies.append(metrics["accuracy"])
    val_precisions.append(metrics["precision"])
    val_recalls.append(metrics["recall"])
    val_f1s.append(metrics["f1"])

    save_path_i = f"{config['train_classifier']['save_path']}_model{i}.pt"
    torch.save(model.state_dict(), save_path_i)


for i in range(n_models):
    model = Classifier(input_dim=embeddings.shape[1], num_classes=7)
    model.load_state_dict(torch.load(f"{config['train_classifier']['save_path']}_model{i}.pt"))
    model.to(device)
    model.eval()
    ensemble_models.append(model)

ensemble = EnsembleClassifier(ensemble_models).to(device)
ensemble.eval()

correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in test_loader_cls:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = ensemble(x_batch)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

test_acc = correct / total
print(f"\nTest Accuracy (Ensemble): {test_acc:.4f}")
# experiment.log_metric("ensemble_test_accuracy", test_acc)

model_names = [f"Model {i+1}" for i in range(n_models)]

def log_bar_chart(metric_values, metric_name):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=model_names, y=metric_values, palette="viridis")
    plt.title(f"{metric_name} Comparison Across Models")
    plt.ylabel(metric_name)
    plt.ylim(0, 1)
    plt.tight_layout()
    path = f"./reconstructions/{metric_name.lower()}_bar_chart.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    experiment.log_image(path)
    plt.close()

log_bar_chart(val_accuracies, "Accuracy")
log_bar_chart(val_precisions, "Precision")
log_bar_chart(val_recalls, "Recall")
log_bar_chart(val_f1s, "F1 Score")
