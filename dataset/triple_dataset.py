from torch.utils.data import Dataset
import random
import os
from PIL import Image
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader

class SkinLesionTripletDataset(Dataset):
    def __init__(self, dataframe, image_dirs, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dirs = image_dirs
        self.transform = transform

        self.label_map = {label: idx for idx, label in enumerate(sorted(self.df['dx'].unique()))}
        self.df['label'] = self.df['dx'].map(self.label_map)

        # Raggruppa gli indici per classe
        self.label_to_indices = {
            label: self.df[self.df['label'] == label].index.tolist()
            for label in self.df['label'].unique()
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor_row = self.df.iloc[idx]
        anchor_img = self._load_image(anchor_row['image_id'])
        anchor_label = anchor_row['label']

        # Trova un positivo diverso dall'anchor
        positive_idx = idx
        while positive_idx == idx:
            positive_idx = random.choice(self.label_to_indices[anchor_label])
        positive_img = self._load_image(self.df.iloc[positive_idx]['image_id'])

        # Trova un negativo (classe diversa)
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = random.choice(list(self.label_to_indices.keys()))
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_img = self._load_image(self.df.iloc[negative_idx]['image_id'])

        return anchor_img, positive_img, negative_img

    def _load_image(self, image_id):
        for directory in self.image_dirs:
            path = os.path.join(directory, image_id + ".jpg")
            if os.path.exists(path):
                image = Image.open(path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image
        raise FileNotFoundError(f"Image {image_id}.jpg not found")
    
def get_triplet_dataloader(config_path="config.yml"):
    # Caricamento configurazione
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    image_dirs = config['data']['image_dirs']
    metadata_path = config['data']['metadata_path']
    image_size = config['data']['image_size']
    val_split = config['data']['val_split']
    batch_size = config['train_autoencoder']['batch_size']
    num_workers = config['data']['num_workers']
    seed = config['data']['seed']

    # Carica il dataframe
    df = pd.read_csv(metadata_path)

    # Filtra le immagini effettivamente presenti
    valid_ids = [img_id for img_id in df['image_id'] if any(os.path.exists(os.path.join(d, img_id + ".jpg")) for d in image_dirs)]
    df = df[df['image_id'].isin(valid_ids)]

    # Split in training e validazione
    train_df, val_df = train_test_split(df, test_size=val_split, stratify=df['dx'], random_state=seed)

    # Trasformazioni standard
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # Crea dataset triplet e dataloader
    train_triplet_dataset = SkinLesionTripletDataset(train_df, image_dirs, transform)
    train_triplet_loader = DataLoader(train_triplet_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_triplet_loader, train_df
