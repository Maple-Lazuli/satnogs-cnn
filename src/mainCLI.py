import argparse
import os
import torch

from data_utils.satnogs_data_manager import SatnogsDataManager
from models.resnet import ResNet, ResBlock, ResBottleneckBlock

from ml_infrastructure.model import Model
from ml_infrastructure.manager import Manager


def main(flags):
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpus

    dm = SatnogsDataManager(batch_size=50).dm

    model1 = Model(net=ResNet(1, ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=1), name='resnet-18')
    model1.criterion = torch.nn.BCEWithLogitsLoss()

    model2 = Model(net=ResNet(1, ResBlock, [3, 4, 6, 3], useBottleneck=False, outputs=1), name='resnet-34')
    model2.criterion = torch.nn.BCEWithLogitsLoss()

    model3 = Model(net=ResNet(1, ResBottleneckBlock, [3, 4, 6, 3], useBottleneck=True, outputs=1), name='resnet-50')
    model3.criterion = torch.nn.BCEWithLogitsLoss()

    manager = Manager(models=[model1, model2, model3], data_manager=dm, epochs=1, start_watcher_app=flags.start_watcher,
                      ip=flags.watcher_ip, port=flags.watcher_port)
    manager.perform()
    manager.save_watcher_results(save_location='./results', save_name='Resnet.json')

    if flags.stop_watcher_on_end:
        manager.shutdown_watcher()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpus', type=str,
                        default='',
                        help='Convert the psd to grey scale.')

    parser.add_argument('--start-watcher', type=bool,
                        default=True,
                        help='Boolean to start a watcher')

    parser.add_argument('--stop-watcher-on-end', type=bool,
                        default=False,
                        help='Boolean to stop the watcher after training')

    parser.add_argument('--watcher-ip', type=str,
                        default='0.0.0.0',
                        help='The IP to use for the watcher')

    parser.add_argument('--watcher-port', type=int,
                        default=5123,
                        help='The port to use for the watcher')

    parser.add_argument('--training-noise', type=bool,
                        default=True,
                        help='Boolean to indicate whether to add noise during training')

    parser.add_argument('--stats', type=str,
                        default='./satnogs-data/stats.json',
                        help='The location of the stats.json file.')

    parsed_flags, _ = parser.parse_known_args()

    main(parsed_flags)
