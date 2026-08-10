"""Dataset de retinopatia diabetica: lee CSV (image, level) + directorio de imagenes."""
import os
import re

import cv2
import pandas as pd
from torch.utils.data import Dataset


class DRDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, transform=None,
                 img_ext: str = '.jpg', subset: int = 0, seed: int = 42):
        df = pd.read_csv(csv_path)
        if subset and subset > 0:
            df = df.sample(n=min(subset, len(df)), random_state=seed).reset_index(drop=True)
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.img_ext = img_ext

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # el CSV puede traer la extension (p.ej. Messidor): normalizar
        name = re.sub(r'(?i)\.(jpe?g|png)$', '', str(row['image']))
        path = os.path.join(self.img_dir, name + self.img_ext)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        label = int(row['level'])
        return img, label
