import torch
import pandas as pd
from torch.utils.data import Dataset
from utils import preprocess_image, transform
from config import SEQ_LENGTH

CHARS = "0123456789"

def encode_label(label):
    return [CHARS.index(c) for c in label]

class CaptchaDataset(Dataset):
    def __init__(self, csv_path, split):
        data = pd.read_csv(csv_path, dtype={'solution': str})
        self.data = data[data['image_path'].str.contains(split)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = preprocess_image(row['image_path'])
        img = transform(img)
        label = torch.tensor(encode_label(row['solution']), dtype=torch.long)
        return img, label
