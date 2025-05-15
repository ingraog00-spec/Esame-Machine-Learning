from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision
import torch
import io
from collections import Counter
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

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
