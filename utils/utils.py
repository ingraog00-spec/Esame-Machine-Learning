from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision
import torch
from tqdm import tqdm

def stratified_split(df, val_size, test_size, seed):
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['dx'],
        random_state=seed
    )

    val_relative_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df['dx'],
        random_state=seed
    )
    return train_df, val_df, test_df

def show_batch_images(images, labels, label_map, title=""):
    grid_img = torchvision.utils.make_grid(images[:16], nrow=4, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img.permute(1, 2, 0))
    label_names = [label_map[l.item()] for l in labels[:16]]
    plt.title(f"{title}\n" + ", ".join(label_names))
    plt.axis("off")
    plt.show()

def plot_class_distribution(loader, label_map, title="Distribuzione Classi"):
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
    plt.show()