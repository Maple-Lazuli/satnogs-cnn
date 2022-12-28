from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


class SatnogsDataset(Dataset):

    def __init__(self, csv, mu=0, sigma=1):
        self.annotations = pd.read_csv(csv)
        self.mu = mu
        self.sigma = sigma

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        example = self.annotations.iloc[index]
        img = (np.fromfile(example['waterfall_location'], dtype=np.uint8).reshape(-1, 623) - self.mu) / self.sigma
        img = torch.from_numpy(img)
        img = img.unsqueeze(0).repeat(1, 1, 1)
        img = img.to(torch.float32)

        target = torch.tensor(example['status']).unsqueeze(0).type(torch.float32)

        return img, target


if __name__ == '__main__':
    train_data = SatnogsDataset("/home/maple/CodeProjects/satnogs_cnn/satnogs-data/train.csv")

    print(train_data[0])
