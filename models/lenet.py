"""
Source: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
This is the example CNN taken from the tutorial for pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_infrastructure.model import Model
from ml_infrastructure.manager import Manager
from ml_infrastructure.data_manager import DataManager
from data_utils.dataset import SatnogsDataset


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 7)
        self.fc1 = nn.Linear(920496, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # Create the data manager

    batch_size = 2
    train_set = SatnogsDataset(csv="../data/train.csv")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    test_set = SatnogsDataset(csv="../data/test.csv")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    val_set = SatnogsDataset(csv="../data/val.csv")
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    classes = ("no-signal", "signal")
    dm = DataManager(train_loader=train_loader, validation_loader=val_loader, test_loader=test_loader, classes=classes)

    model = Model(net=Net().cuda(), name='lenet')
    model.criterion = torch.nn.BCEWithLogitsLoss()
    manager = Manager(models=[model], data_manager=dm, epochs=1)
    manager.perform()
    manager.save_watcher_results(save_location='../results', save_name='Lenet.json')

    manager.shutdown_watcher()
