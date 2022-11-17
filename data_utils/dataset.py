from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor


class SatnogsDataset(Dataset):

    def __init__(self, csv):
        self.annotations = pd.read_csv(csv)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        example = self.annotations.iloc[index]
        img = np.fromfile(example['waterfall_location'], dtype=np.uint8).reshape(-1, 623)
        img = Image.fromarray(img).resize((1542, 1542), Image.ANTIALIAS)
        return pil_to_tensor(img).type(torch.float), example['status']
