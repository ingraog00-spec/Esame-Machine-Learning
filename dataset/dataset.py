import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

# Dataset personalizzato per immagini di lesioni cutanee
class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, image_dirs, transform=None, minority_transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dirs = image_dirs
        self.transform = transform
        self.minority_transform = minority_transform

        # Mappa le etichette testuali a indici numerici
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.df['dx'].unique()))}
        self.df['label'] = self.df['dx'].map(self.label_map)

        # Identifica le classi minoritarie (meno del 20% rispetto alla più numerosa)
        class_counts = self.df['dx'].value_counts()
        max_count = class_counts.max()
        self.minority_classes = class_counts[class_counts < max_count * 0.2].index.tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Estrae la riga corrispondente
        row = self.df.iloc[idx]

        # Trova il percorso dell'immagine
        image_path = self._find_image_path(row['image_id'])

        # Carica l'immagine e converte in RGB
        image = Image.open(image_path).convert("RGB")

        # Applica trasformazioni diverse per le classi minoritarie
        if row['dx'] in self.minority_classes and self.minority_transform:
            image = self.minority_transform(image)
        elif self.transform:
            image = self.transform(image)

        return image, row['label']

    # Cerca il file immagine nei percorsi specificati
    def _find_image_path(self, image_id):
        for directory in self.image_dirs:
            path = os.path.join(directory, image_id + ".jpg")
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Image {image_id}.jpg non trovata.")

# Calcola i pesi per ciascun campione per il campionamento bilanciato
def compute_sample_weights(dataset):
    labels = dataset.df['label'].values

    # Conta le occorrenze di ciascuna classe
    class_counts = np.bincount(labels)

    # Inverso della frequenza: classi rare hanno peso maggiore
    class_weights = 1. / class_counts

    # Applica i pesi in base alla classe di ogni esempio
    sample_weights = class_weights[labels]
    return sample_weights

# Funzione principale per creare i DataLoader di training e validation
def get_dataloaders(config_path="config.yml"):
    # Carica i parametri da file di configurazione
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Estrazione parametri
    image_dirs = config['data']['image_dirs']
    metadata_path = config['data']['metadata_path']
    image_size = config['data']['image_size']
    val_split = config['data']['val_split']
    batch_size = config['train_autoencoder']['batch_size']
    num_workers = config['data']['num_workers']
    seed = config['data']['seed']

    # Carica il file dei metadati
    print(f"\nCaricamento metadati da: {metadata_path}")
    df = pd.read_csv(metadata_path)

    # Verifica che le immagini esistano nei percorsi specificati
    valid_ids = []
    for image_id in df['image_id']:
        if any(os.path.exists(os.path.join(d, image_id + ".jpg")) for d in image_dirs):
            valid_ids.append(image_id)
    df = df[df['image_id'].isin(valid_ids)]

    # Suddivide il dataset in training e validation in modo stratificato
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        stratify=df['dx'],
        random_state=seed
    )

    print(f"\nSuddivisione dataset:")
    print(f"- Train set: {len(train_df)}")
    print(f"- Validation set: {len(val_df)}")

    # Trasformazioni base da applicare a tutte le immagini
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # Trasformazioni aggiuntive per le classi minoritarie (data augmentation)
    minority_transform = transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20), interpolation=Image.BICUBIC),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # Crea i dataset
    train_dataset = SkinLesionDataset(train_df, image_dirs, transform=transform, minority_transform=minority_transform)
    val_dataset = SkinLesionDataset(val_df, image_dirs, transform=transform)

    # Calcola i pesi per il campionamento bilanciato
    sample_weights = compute_sample_weights(train_dataset)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Crea i DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"- Batch size: {batch_size}")
    print(f"- Numero workers: {num_workers}")
    print("DataLoader pronti.\n")

    return train_loader, val_loader

# Funzione per creare il DataLoader del test set
def get_test_dataloader(test_image_dir, test_metadata_path, image_size, batch_size, num_workers):
    # Carica metadati test
    df_test = pd.read_csv(test_metadata_path)

    # Filtra solo immagini realmente esistenti
    valid_ids = [img_id for img_id in df_test['image_id'] if os.path.exists(os.path.join(test_image_dir, img_id + ".jpg"))]
    df_test = df_test[df_test['image_id'].isin(valid_ids)]

    # Trasformazioni per il test
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # Crea dataset e dataloader
    test_dataset = SkinLesionDataset(df_test, [test_image_dir], transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader
