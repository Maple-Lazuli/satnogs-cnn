import torch
import torchvision.transforms as transforms
from data_utils.dataset import SatnogsDataset


class Loader:
    def __init__(self, location="../data", batch_size=1):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.train_set = SatnogsDataset(csv="./data/train.csv")

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size,
                                                        shuffle=True, num_workers=2)

        self.train_set = SatnogsDataset(csv="./data/test.csv")

        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size,
                                                       shuffle=False, num_workers=2)

        self.classes = (0, 1)
