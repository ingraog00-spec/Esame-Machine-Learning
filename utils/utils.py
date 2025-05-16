import matplotlib.pyplot as plt
import torchvision
import torch
import io
from collections import Counter
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def show_batch_images(images, labels, label_map, title="", experiment=None):
    grid_img = torchvision.utils.make_grid(images[:16], nrow=4, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img.permute(1, 2, 0))
    label_names = [label_map[l.item()] for l in labels[:16]]
    plt.title(f"{title}\n" + ", ".join(label_names))
    plt.axis("off")
    
    if experiment:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        experiment.log_image(buf, name=f"{title.replace(' ', '_')}.png")
        buf.close()
    plt.show()
    plt.close()

def plot_class_distribution(loader, label_map, title="Distribuzione Classi", experiment=None):
    label_counts = torch.zeros(len(label_map), dtype=torch.int32)
    for _, labels in loader:
        for l in labels:
            label_counts[l] += 1

    labels = list(label_map.values())
    plt.figure(figsize=(10, 4))
    plt.bar(labels, label_counts.tolist())
    plt.title(title)
    plt.xlabel("Classi")
    plt.ylabel("Frequenza")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if experiment:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        experiment.log_image(buf, name=f"{title.replace(' ', '_')}.png")
        buf.close()
    plt.show()
    plt.close()

def count_class_distribution(loader):
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.tolist())
    return Counter(all_labels)

def log_class_counts_per_split(train_loader, val_loader, test_loader, inv_label_map, experiment):
    for split_name, loader in zip(['Train', 'Validation', 'Test'], [train_loader, val_loader, test_loader]):
        counts = count_class_distribution(loader)
        print(f"\n{split_name} Set Distribution:")
        for class_id, count in sorted(counts.items()):
            label_name = inv_label_map[class_id]
            print(f"  {label_name} ({class_id}): {count}")
            experiment.log_metric(f"{split_name}_count_{label_name}", count)

def tsne_visualization(embeddings, labels, inv_label_map, experiment, title="t-SNE of Latent Space"):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    df = pd.DataFrame()
    df["x"] = reduced[:, 0]
    df["y"] = reduced[:, 1]
    df["label"] = [inv_label_map[l] for l in labels]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette="tab10", alpha=0.7)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    save_path = "./images/tsne_latent_space.png"
    plt.savefig(save_path)
    experiment.log_image(save_path, name="TSNE Latent Space")
    plt.close()

def generate_graphics(test_loader_cls, device, ensemble, val_scores, test_results, class_names, experiment, n_models):
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
    plt.close()

    # GRAFICO 2: MATRICE DI CONFUSIONE
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format=".0f")
    plt.title("Matrice di Confusione - Ensemble")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    experiment.log_image("confusion_matrix.png")
    plt.close()

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
    plt.close()