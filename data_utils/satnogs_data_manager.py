import json
from dataclasses import dataclass

import torch
from ml_infrastructure.data_manager import DataManager

from data_utils.dataset import SatnogsDataset


@dataclass
class SatnogsDataManager:
    batch_size: int = 5

    def __post_init__(self):

        with open('./satnogs-data/stats.json', 'r') as file_in:
            stats = json.load(file_in)

        train_set = SatnogsDataset(csv="./satnogs-data/train.csv", mu=stats['mu'], sigma=stats['sigma'])
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size,
                                                        shuffle=True, num_workers=2)
        test_set = SatnogsDataset(csv="./satnogs-data/test.csv", mu=stats['mu'], sigma=stats['sigma'])
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size,
                                                       shuffle=False, num_workers=2)
        val_set = SatnogsDataset(csv="./satnogs-data/val.csv", mu=stats['mu'], sigma=stats['sigma'])
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=2)
        self.classes = ("no-signal", "signal")

        self.dm = DataManager(train_loader=self.train_loader, validation_loader=self.val_loader,
                           test_loader=self.test_loader, classes=self.classes)
