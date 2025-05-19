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
    # Visualizza una griglia di immagini con le rispettive etichette
    grid_img = torchvision.utils.make_grid(images[:16], nrow=4, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img.permute(1, 2, 0))
    label_names = [label_map[l.item()] for l in labels[:16]]
    plt.title(f"{title}\n" + ", ".join(label_names))
    plt.axis("off")
    
    # Salva e logga l'immagine
    if experiment:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        experiment.log_image(buf, name=f"{title.replace(' ', '_')}.png")
        buf.close()
    plt.show()
    plt.close()

def plot_class_distribution(loader, label_map, title="Distribuzione Classi", experiment=None):
    # Calcola la distribuzione delle classi in un DataLoader
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

    # Salva e logga il grafico
    if experiment:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        experiment.log_image(buf, name=f"{title.replace(' ', '_')}.png")
        buf.close()
    plt.show()
    plt.close()

def count_class_distribution(loader):
    # Conta la frequenza di ciascuna classe in un DataLoader
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.tolist())
    return Counter(all_labels)

def log_class_counts_per_split(train_loader, val_loader, test_loader, inv_label_map, experiment):
    # Logga la distribuzione delle classi per ogni split (train, validation, test)
    for split_name, loader in zip(['Train', 'Validation', 'Test'], [train_loader, val_loader, test_loader]):
        counts = count_class_distribution(loader)
        print(f"\n{split_name} Set Distribution:")
        for class_id, count in sorted(counts.items()):
            label_name = inv_label_map[class_id]
            print(f"  {label_name} ({class_id}): {count}")
            experiment.log_metric(f"{split_name}_count_{label_name}", count)

def print_section(title):
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60 + "\n")


def tsne_visualization(embeddings, labels, inv_label_map, experiment, title="t-SNE of Latent Space"):
    # -- Visualizza le embedding tramite t-SNE per riduzione dimensionale --
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
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

    # Salva e logga il grafico t-SNE
    save_path = "./images/tsne_latent_space.png"
    plt.savefig(save_path)
    experiment.log_image(save_path, name="TSNE Latent Space")
    plt.close()

def generate_graphics(test_loader_cls, device, model, class_names, experiment):
    # Genera grafici di valutazione sul test set: matrice di confusione e metriche per classe
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader_cls:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Matrice di confusione
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format=".0f")
    plt.title("Matrice di Confusione - Test Set")
    plt.tight_layout()
    plt.savefig("./images/confusion_matrix.png")
    experiment.log_image("./images/confusion_matrix.png")
    plt.close()

    # precision, recall, f1-score per ogni classe
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
    plt.title('Performance per Classe - Test Set')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./images/class_metrics.png")
    experiment.log_image("./images/class_metrics.png")
    plt.close()