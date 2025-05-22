import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import yaml
from sklearn.model_selection import train_test_split

# Dataset personalizzato per immagini di Skin Lesion
class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, image_dirs, transform=None, minority_transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dirs = image_dirs
        self.transform = transform
        self.minority_transform = minority_transform
        # Mappa ogni etichetta a un intero
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.df['dx'].unique()))}
        self.df['label'] = self.df['dx'].map(self.label_map)

        # Identifica le classi minoritarie (meno del 20% rispetto alla classe più numerosa)
        class_counts = self.df['dx'].value_counts()
        max_count = class_counts.max()
        self.minority_classes = class_counts[class_counts < max_count * 0.2].index.tolist()

    def __len__(self):
        # Restituisce il numero totale di campioni
        return len(self.df)

    def __getitem__(self, idx):
        # Restituisce l'immagine e la sua etichetta corrispondente
        row = self.df.iloc[idx]
        image_path = self._find_image_path(row['image_id'])

        image = Image.open(image_path).convert("RGB")

        # Applica trasformazioni diverse per le classi minoritarie se specificato
        if row['dx'] in self.minority_classes and self.minority_transform:
            image = self.minority_transform(image)
        elif self.transform:
            image = self.transform(image)

        return image, row['label']

    def _find_image_path(self, image_id):
        # Cerca l'immagine in tutte le directory fornite
        for directory in self.image_dirs:
            path = os.path.join(directory, image_id + ".jpg")
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Image {image_id}.jpg non trovata in nessuna directory")

# Funzione per creare i dataloader di train e validation
def get_dataloaders(config_path="config.yml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    image_dirs = config['data']['image_dirs']
    metadata_path = config['data']['metadata_path']
    image_size = config['data']['image_size']
    val_split = config['data']['val_split']
    batch_size = config['train_autoencoder']['batch_size']
    num_workers = config['data']['num_workers']
    seed = config['data']['seed']

    df = pd.read_csv(metadata_path)

    # Filtra le immagini che non esistono nelle directory
    valid_ids = []
    for image_id in df['image_id']:
        if any(os.path.exists(os.path.join(d, image_id + ".jpg")) for d in image_dirs):
            valid_ids.append(image_id)
    df = df[df['image_id'].isin(valid_ids)]

    # Suddivide il dataset in train e validation, stratificando per etichetta
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        stratify=df['dx'],
        random_state=seed
    )

    # ------------------ OVERSAMPLING DELLE CLASSI MINORITARIE ------------------
    """ class_counts = train_df['dx'].value_counts()
    max_count = class_counts.max()
    minority_classes = class_counts[class_counts < max_count * 0.2].index.tolist()

    oversampled_rows = []
    for cls in minority_classes:
        cls_rows = train_df[train_df['dx'] == cls]
        n_to_add = max_count - len(cls_rows)
        oversampled = cls_rows.sample(n=n_to_add, replace=True, random_state=seed)
        oversampled_rows.append(oversampled)

    if oversampled_rows:
        train_df = pd.concat([train_df] + oversampled_rows).reset_index(drop=True)
 """

    # Trasformazione standard per tutte le immagini
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # Trasformazioni aggiuntive per le classi minoritarie (data augmentation)
    minority_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # Crea gli oggetti Dataset
    train_dataset = SkinLesionDataset(train_df, image_dirs, transform=transform, minority_transform=minority_transform)
    val_dataset = SkinLesionDataset(val_df, image_dirs, transform)

    # Crea i DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

# Funzione per creare il dataloader di test
def get_test_dataloader(test_image_dir, test_metadata_path, image_size, batch_size, num_workers):
    df_test = pd.read_csv(test_metadata_path)

    # Filtra le immagini mancanti
    valid_ids = [img_id for img_id in df_test['image_id'] if os.path.exists(os.path.join(test_image_dir, img_id + ".jpg"))]
    df_test = df_test[df_test['image_id'].isin(valid_ids)]

    # Trasformazione per le immagini di test
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # Crea Dataset e DataLoader per il test set
    test_dataset = SkinLesionDataset(df_test, [test_image_dir], transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader
