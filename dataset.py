import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils import stratified_split
import yaml

class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, image_dirs, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dirs = image_dirs
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.df['dx'].unique()))}
        self.df['label'] = self.df['dx'].map(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self._find_image_path(row['image_id'])

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, row['label']

    def _find_image_path(self, image_id):
        for directory in self.image_dirs:
            path = os.path.join(directory, image_id + ".jpg")
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Image {image_id}.jpg not found in any image directory")

def get_dataloaders(config_path="config.yml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    image_dirs = config['data']['image_dirs']
    metadata_path = config['data']['metadata_path']
    image_size = config['data']['image_size']
    val_split = config['data']['val_split']
    test_split = config['data']['test_split']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    seed = config['training']['seed']

    df = pd.read_csv(metadata_path)

    valid_ids = []
    for image_id in df['image_id']:
        if any(os.path.exists(os.path.join(d, image_id + ".jpg")) for d in image_dirs):
            valid_ids.append(image_id)
    df = df[df['image_id'].isin(valid_ids)]

    train_df, val_df, test_df = stratified_split(df, val_split, test_split, seed)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    train_dataset = SkinLesionDataset(train_df, image_dirs, transform)
    val_dataset = SkinLesionDataset(val_df, image_dirs, transform)
    test_dataset = SkinLesionDataset(test_df, image_dirs, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
