from dataset import get_dataloaders
from utils import show_batch_images, plot_class_distribution
from train import train_autoencoder
from models import ConvAutoencoder
import torch
import yaml
from comet_ml import Experiment

if __name__ == "__main__":
    experiment = Experiment()
    experiment.set_name("Autoencoder_Skin_Lesion")
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, test_loader = get_dataloaders("config.yml")

    label_map = train_loader.dataset.label_map
    inv_label_map = {v: k for k, v in label_map.items()}

    """ print("- Visualizzo un batch dal train loader")
    images, labels = next(iter(train_loader))
    show_batch_images(images, labels, inv_label_map, title="Batch di Training")

    print("- Distribuzione delle classi nel training set:")
    plot_class_distribution(train_loader, inv_label_map, title="Distribuzione Classi - Train")

    print("- Distribuzione delle classi nel validation set:")
    plot_class_distribution(val_loader, inv_label_map, title="Distribuzione Classi - Validation")

    print("- Distribuzione delle classi nel test set:")
    plot_class_distribution(test_loader, inv_label_map, title="Distribuzione Classi - Test") """

    autoencoder = ConvAutoencoder(encoded_space_dim=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_autoencoder(autoencoder, train_loader, config, device, experiment)
