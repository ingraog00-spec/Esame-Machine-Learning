import comet_ml
import torch
from torch.utils.data import TensorDataset, DataLoader
from comet_ml import Experiment
import yaml
from models.model_classifier import Classifier, EnsembleClassifier
from utils.train_classifier import train_classifier
from utils.test import test_classifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

experiment = Experiment()
experiment.set_name("Prova del main 2")

with open("config.yml") as f:
    config = yaml.safe_load(f)

default_class_names = [str(i) for i in range(7)]
class_names = config.get("class_names", default_class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = torch.load("./save_model_embeddings/embeddings.pt", weights_only=False)
train_embeddings, train_labels = data["train"]
val_embeddings, val_labels = data["val"]
test_embeddings, test_labels = data["test"]

train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
test_labels = torch.tensor(test_labels)

train_dataset = TensorDataset(train_embeddings, train_labels)
val_dataset = TensorDataset(val_embeddings, val_labels)
test_dataset = TensorDataset(test_embeddings, test_labels)

train_loader_cls = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader_cls = DataLoader(val_dataset, batch_size=32)
test_loader_cls = DataLoader(test_dataset, batch_size=32)

n_models = config["train_classifier"]["n_models"]
save_base_path = config["train_classifier"]["save_path"]

skf = StratifiedKFold(n_splits=n_models, shuffle=True, random_state=42)
ensemble_models = []
val_scores = []

# TRAINING DEI MODELLI
for i, (train_idx, val_idx) in enumerate(skf.split(train_embeddings, train_labels)):
    print(f"\n--- Training fold {i+1}/{n_models} ---")

    model = Classifier(input_dim=256, num_classes=7)
    metrics = train_classifier(model, train_loader_cls, val_loader_cls, config, device, experiment, model_name=f"Modello_{i+1}")

    torch.save(model.state_dict(), f"{save_base_path}_fold{i}.pt")
    val_scores.append(metrics)

    print(f"Model {i+1} | Acc: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f}")

# CARICAMENTO MODELLI E CREAZIONE ENSEMBLE
for i in range(n_models):
    model = Classifier(input_dim=256, num_classes=7)
    model.load_state_dict(torch.load(f"{save_base_path}_fold{i}.pt"))
    model.to(device)
    model.eval()
    ensemble_models.append(model)

ensemble = EnsembleClassifier(ensemble_models).to(device)

# TEST DELL’ENSEMBLE
test_results = test_classifier(
    model=ensemble,
    data_loader=test_loader_cls,
    device=device,
    experiment=experiment,
    title="Test Ensemble",
    log_prefix="ensemble_test_"
)

# PREDIZIONI PER GRAFICI
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader_cls:
        inputs = inputs.to(device)
        outputs = ensemble(inputs)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# GRAFICO 1: ACCURACY & F1 DEI MODELLI E DELL’ENSEMBLE
model_ids = [f"Model {i+1}" for i in range(n_models)] + ["Ensemble"]
accs = [m['accuracy'] for m in val_scores] + [test_results["accuracy"]]
f1s = [m['f1'] for m in val_scores] + [test_results["f1"]]

plt.figure(figsize=(8, 5))
plt.plot(model_ids, accs, marker='o', label='Accuracy')
plt.plot(model_ids, f1s, marker='o', label='F1-score')
plt.title("Confronto Modelli vs Ensemble")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("metrics_comparison.png")
experiment.log_image("metrics_comparison.png")

# GRAFICO 2: MATRICE DI CONFUSIONE
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", values_format=".0f")
plt.title("Matrice di Confusione - Ensemble")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
experiment.log_image("confusion_matrix.png")

# GRAFICO 3: PRECISION/RECALL/F1 PER CLASSE
report = classification_report(all_labels, all_preds, output_dict=True)

precision = [report[str(i)]['precision'] for i in range(len(class_names))]
recall = [report[str(i)]['recall'] for i in range(len(class_names))]
f1 = [report[str(i)]['f1-score'] for i in range(len(class_names))]

x = np.arange(len(class_names))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, f1, width, label='F1-score')
plt.xticks(x, class_names)
plt.ylabel('Score')
plt.title('Performance per Classe - Ensemble')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("class_metrics.png")
experiment.log_image("class_metrics.png")
