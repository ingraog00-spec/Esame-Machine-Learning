import comet_ml
from dataset.dataset import get_dataloaders
from utils.utils import show_batch_images, plot_class_distribution, log_class_counts_per_split
from utils.train_autoencoder import train_autoencoder
from models.models_autoencoder import ConvConditionalVAE
import torch
import yaml
from comet_ml import Experiment
from utils.feature_extraction import extract_embeddings
from utils.train_classifier import train_classifier
from models.model_classifier import Classifier, EnsembleClassifier
from torch.utils.data import DataLoader, TensorDataset
from utils.test import test_classifier
from utils.utils import tsne_visualization
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os

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

if __name__ == "__main__":
    experiment = Experiment()
    experiment.set_name("prova CVAE - completa")
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, test_loader = get_dataloaders("config.yml")

    label_map = train_loader.dataset.label_map
    inv_label_map = {v: k for k, v in label_map.items()}

    """ print("- Visualizzo un batch dal train loader")
    images, labels = next(iter(train_loader))
    show_batch_images(images, labels, inv_label_map, title="Batch di Training", experiment=experiment)

    print("- Distribuzione delle classi nel training set:")
    plot_class_distribution(train_loader, inv_label_map, title="Distribuzione Classi - Train", experiment=experiment)

    print("- Distribuzione delle classi nel validation set:")
    plot_class_distribution(val_loader, inv_label_map, title="Distribuzione Classi - Validation", experiment=experiment)

    print("- Distribuzione delle classi nel test set:")
    plot_class_distribution(test_loader, inv_label_map, title="Distribuzione Classi - Test", experiment=experiment)

    log_class_counts_per_split(train_loader, val_loader, test_loader, inv_label_map, experiment) """

    autoencoder = ConvConditionalVAE(latent_dim=256, num_classes=len(label_map))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_autoencoder(autoencoder, train_loader, config, device, experiment)

    autoencoder.load_state_dict(torch.load(config["train_autoencoder"]["save_path"]))
    autoencoder.to(device)
    autoencoder.eval()

    print("- Estrazione degli embeddings per t-SNE")
    train_embeddings, train_labels = extract_embeddings(autoencoder, train_loader, device)
    tsne_visualization(train_embeddings, train_labels, inv_label_map, experiment, "t-SNE of Train Set")

    val_embeddings, val_labels = extract_embeddings(autoencoder, val_loader, device)
    tsne_visualization(val_embeddings, val_labels, inv_label_map, experiment, "t-SNE of Validation Set")

    test_embeddings, test_labels = extract_embeddings(autoencoder, test_loader, device)
    tsne_visualization(test_embeddings, test_labels, inv_label_map, experiment, "t-SNE of Test Set")

    torch.save({
        "train": (train_embeddings, train_labels),
        "val": (val_embeddings, val_labels),
        "test": (test_embeddings, test_labels)
    }, "./save_model_embeddings/embeddings.pt")

    print("Embeddings salvati in embeddings.pt")

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
    ensemble_models = []

    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    for i in range(n_models):
        print(f"\n--- Training model {i+1}/{n_models} ---")
        model = Classifier(input_dim=256, num_classes=7)

        metrics = train_classifier(model, train_loader_cls, val_loader_cls, config, device, experiment=experiment)

        val_accuracies.append(metrics["accuracy"])
        val_precisions.append(metrics["precision"])
        val_recalls.append(metrics["recall"])
        val_f1s.append(metrics["f1"])

        save_path_i = f"{config['train_classifier']['save_path']}_model{i}.pt"
        torch.save(model.state_dict(), save_path_i)


    for i in range(n_models):
        model = Classifier(input_dim=256, num_classes=7)
        model.load_state_dict(torch.load(f"{config['train_classifier']['save_path']}_model{i}.pt"))
        model.to(device)
        model.eval()
        ensemble_models.append(model)

    ensemble = EnsembleClassifier(ensemble_models).to(device)

    model_names = [f"Model {i+1}" for i in range(n_models)]

    log_bar_chart(val_accuracies, "Accuracy")
    log_bar_chart(val_precisions, "Precision")
    log_bar_chart(val_recalls, "Recall")
    log_bar_chart(val_f1s, "F1 Score")

    test_classifier(model = ensemble, data_loader = test_loader_cls, device = device, experiment = experiment, title = "Test Ensemble", log_prefix = "ensemble_test_")
