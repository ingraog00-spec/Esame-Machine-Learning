import os
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from vision_embeddings import VisionEmbeddings

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder, filename)
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                filenames.append(filename)
            except Exception as e:
                print(f"Errore con immagine {filename}: {e}")
    return images, filenames

def main():
    folders = [
        "./dataverse_files/HAM10000_images_part_1",
        "./dataverse_files/HAM10000_images_part_2"
    ]

    metadata_path = "./dataverse_files/HAM10000_metadata"
    metadata = pd.read_csv(metadata_path)
    image_to_label = dict(zip(metadata["image_id"], metadata["dx"]))

    embedder = VisionEmbeddings(device='cpu')

    dataset_rows = []

    for folder in folders:
        print(f"\nProcessing folder: {folder}")
        images, filenames = load_images_from_folder(folder)

        for img, fname in tqdm(zip(images, filenames), total=len(images)):
            image_id = os.path.splitext(fname)[0]
            label = image_to_label.get(image_id)
            if label is None:
                print(f"Label mancante per: {image_id}, saltato.")
                continue

            features = embedder.extract(img).flatten()
            row = list(features) + [fname, label]
            dataset_rows.append(row)

    num_features = len(dataset_rows[0]) - 2
    columns = [f"feature_{i}" for i in range(num_features)] + ["filename", "label"]
    df = pd.DataFrame(dataset_rows, columns=columns)

    df.to_csv("features_skin_lesion.csv", index=False)
    print("\nDataset salvato in: features_skin_lesion.csv")

if __name__ == "__main__":
    main()
