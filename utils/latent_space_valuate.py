import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import torch
torch.set_num_threads(4)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

def evaluate_latent_space(model, dataloader, device, experiment=None, epoch=None, save_dir="./images/latent_space"):
    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            lbls = lbls.to(device)
            _, mu, _, _ = model(images, lbls)  # Mu = media latente
            latents.append(mu.cpu().numpy())
            labels.extend(lbls.cpu().numpy())

    latents = np.concatenate(latents)
    labels = np.array(labels)

    # Normalizzazione
    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)

    # ------------------- CLUSTERING K-MEANS -------------------
    n_clusters = 7
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_preds = kmeans.fit_predict(latents_scaled)

    # ------------------- SILHOUETTE -------------------
    sil_score = silhouette_score(latents_scaled, cluster_preds)

    # ------------------- VISUALIZZAZIONE PCA -------------------
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("tab20", n_colors=len(np.unique(cluster_preds)))

    scatter = ax.scatter(
        latents_2d[:, 0], latents_2d[:, 1],
        c=cluster_preds,
        cmap=matplotlib.colors.ListedColormap(palette),
        s=60,
        edgecolors='k',
        linewidths=0.3,
        alpha=0.75
    )

    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title(f"Spazio Latente (KMeans) - Epoch {epoch if epoch is not None else ''}", fontsize=16)
    ax.set_xlabel("PCA 1", fontsize=12)
    ax.set_ylabel("PCA 2", fontsize=12)

    handles, labels_ = scatter.legend_elements()
    legend = ax.legend(handles, [f"Cluster {i}" for i in range(len(handles))], 
                    title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.add_artist(legend)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"latent_epoch_{epoch}.png" if epoch else "latent_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------- LOG SU CONSOLE -------------------
    print(f"\n[KMEANS {epoch}] Silhouette Score: {sil_score:.4f}")

    # ------------------- LOG SU COMET -------------------
    if experiment:
        experiment.log_metric("latent_silhouette", sil_score, step=epoch)
        experiment.log_image(plot_path, name=f"latent_space_epoch_{epoch}")

    return sil_score