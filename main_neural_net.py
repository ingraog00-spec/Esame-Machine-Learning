import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
import comet_ml
import torch
torch.set_num_threads(4)
from dataset.dataset import get_dataloaders, get_test_dataloader
from utils.utils import show_batch_images, plot_class_distribution, log_class_counts_per_split, generate_graphics, tsne_visualization, print_section
from utils.train_autoencoder import train_autoencoder
from models.models_autoencoder import ConvConditionalVAE
import yaml
from comet_ml import Experiment
from utils.feature_extraction import extract_embeddings
from utils.train_classifier import train_classifier
from models.model_classifier import Classifier
from torch.utils.data import DataLoader, TensorDataset
from utils.test import test_classifier

if __name__ == "__main__":
    print_section("Inizio Esperimento")

    # Inizializzazione dell'esperimento su Comet.ml per il tracking automatico dei risultati
    experiment = Experiment()

    # Simulazione Autoencoder e Classificatore Skin Lesion Classification
    experiment.set_name("test")

    # Caricamento del file di configurazione
    print_section("Caricamento Configurazione")
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    
    # Creazione dei DataLoader per training e validazione
    print_section("Preparazione Dataset e DataLoader")
    train_loader, val_loader = get_dataloaders("config.yml")

    # Estrazione della mappa delle label da indice a nome
    label_map = train_loader.dataset.label_map
    inv_label_map = {v: k for k, v in label_map.items()}

    # Caricamento del DataLoader per il test set
    test_loader = get_test_dataloader(
        test_image_dir="./dataverse_files/ISIC2018_Task3_Test_Images",
        test_metadata_path="./dataverse_files/ISIC2018_Task3_Test_GroundTruth.csv",
        image_size=config['data']['image_size'],
        batch_size=config['train_autoencoder']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    """  # Visualizzazione di un batch di immagini e relative label dal training
    print_section("Visualizzazione Batch Immagini")
    images, labels = next(iter(train_loader))
    show_batch_images(images, labels, inv_label_map, title="Batch di Training", experiment=experiment)

    print_section("Distribuzione Classi nei Dataset")
    # Analisi e visualizzazione della distribuzione delle classi nel training, validation, test set
    print("- Training Set")
    plot_class_distribution(train_loader, inv_label_map, title="Distribuzione Classi - Train", experiment=experiment)

    print("- Validation Set")
    plot_class_distribution(val_loader, inv_label_map, title="Distribuzione Classi - Validation", experiment=experiment)

    print("- Test Set")
    plot_class_distribution(test_loader, inv_label_map, title="Distribuzione Classi - Test", experiment=experiment)

    # Log dettagliato dei conteggi degli split del dataset
    log_class_counts_per_split(train_loader, val_loader, test_loader, inv_label_map, experiment) """

    print_section("Inizializzazione Autoencoder")

    # Inizializzazione del modello autoencoder condizionale convoluzionale,
    # usato per l'estrazione di features latenti (embedding) dalle immagini
    autoencoder = ConvConditionalVAE(latent_dim=config['latent_dim'], num_classes=len(label_map))

    # Scelta del device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device utilizzato: {device}")

    # Training del modello autoencoder con i dati di training
    print_section("Training Autoencoder")
    train_autoencoder(autoencoder, train_loader, config, device, experiment)
    print("Autoencoder allenato. Caricamento pesi salvati...")

    # Caricamento dei pesi salvati dopo il training, preparazione per estrazione embeddings
    autoencoder.load_state_dict(torch.load(config["train_autoencoder"]["save_path"]))
    autoencoder.to(device)
    autoencoder.eval()    # Modalità evaluation: disabilita dropout e batchnorm

    # Estrazione degli embeddings latenti dal modello autoencoder su train, val e test set
    print_section("Estrazione e Visualizzazione t-SNE Embeddings")
    train_embeddings, train_labels = extract_embeddings(autoencoder, train_loader, device)
    tsne_visualization(train_embeddings, train_labels, inv_label_map, experiment, "t-SNE of Train Set")

    val_embeddings, val_labels = extract_embeddings(autoencoder, val_loader, device)
    tsne_visualization(val_embeddings, val_labels, inv_label_map, experiment, "t-SNE of Validation Set")

    test_embeddings, test_labels = extract_embeddings(autoencoder, test_loader, device)
    tsne_visualization(test_embeddings, test_labels, inv_label_map, experiment, "t-SNE of Test Set")

    # Salvataggio degli embeddings estratti
    print("Salvataggio embeddings...")
    torch.save({
        "train": (train_embeddings, train_labels),
        "val": (val_embeddings, val_labels),
        "test": (test_embeddings, test_labels)
    }, f"{config['test_classifier']['embeddings_path']}")

    # Conversione delle label in tensori
    print_section("Preparazione Dati per Classificatore")
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)
    test_labels = torch.tensor(test_labels)

    # Creazione dei dataset combinando embeddings e label
    train_dataset = TensorDataset(train_embeddings, train_labels)
    val_dataset = TensorDataset(val_embeddings, val_labels)
    test_dataset = TensorDataset(test_embeddings, test_labels)

    # Presa la dimensione del batch dal file di configurazione
    batch_size_cls = config["train_classifier"]["batch_size"]

    # Creazione dei DataLoader per training, validazione e test del classificatore
    train_loader_cls = DataLoader(train_dataset, batch_size=batch_size_cls, shuffle=True)
    val_loader_cls = DataLoader(val_dataset, batch_size=batch_size_cls)
    test_loader_cls = DataLoader(test_dataset, batch_size=batch_size_cls)

    save_base_path = config["train_classifier"]["save_path"]

    # Inizializzazione del modello classificatore
    print_section("Training Classificatore")
    model = Classifier(input_dim=config['latent_dim'], num_classes=7)
    model.to(device)

    # Training del classificatore sui dati di embedding
    metrics = train_classifier(model, train_loader_cls, val_loader_cls, config, device, experiment)

    # Salvataggio del modello classificatore
    print("Salvataggio modello classificatore...")
    torch.save(model.state_dict(), f"{save_base_path}")

    # Messa in modalità eval per test
    print_section("Valutazione del Classificatore sul Test Set")
    model.eval()

    # Valutazione del modello classificatore sul test set
    test_results = test_classifier(
        model=model,
        data_loader=test_loader_cls,
        device=device,
        experiment=experiment,
        title="Test Classifier")

    # Generazione di grafici per della valutazione finale e metriche di performance
    print_section("Generazione Grafici Finali")
    generate_graphics(test_loader_cls, device, model, inv_label_map, experiment)
    
    print("Fine processo. Tutto completato con successo!")
