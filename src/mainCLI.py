import argparse
import os
import torch

from data_utils.dataset import SatnogsDataset
from models.resnet import ResNet, ResBlock

from ml_infrastructure.data_manager import DataManager
from ml_infrastructure.model import Model
from ml_infrastructure.manager import Manager


def main(flags):
    os.environ["CUDA_AVAILABLE_DEVICES"] = flags.gpus

    batch_size = 50
    train_set = SatnogsDataset(csv="./data/train.csv")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    test_set = SatnogsDataset(csv="./data/test.csv")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    val_set = SatnogsDataset(csv="./data/val.csv")
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    classes = ("no-signal", "signal")
    dm = DataManager(train_loader=train_loader, validation_loader=val_loader, test_loader=test_loader, classes=classes)

    model = Model(net=ResNet(1, ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=1), name='resnet')
    model.criterion = torch.nn.BCEWithLogitsLoss()
    manager = Manager(models=[model], data_manager=dm, epochs=100)
    manager.perform()
    manager.save_watcher_results(save_location='./results', save_name='Resnet.json')

    manager.shutdown_watcher()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpus', type=str,
                        default='',
                        help='Convert the psd to grey scale.')

    parsed_flags, _ = parser.parse_known_args()

    main(parsed_flags)
