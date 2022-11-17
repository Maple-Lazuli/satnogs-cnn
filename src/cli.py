import argparse

from models.get_model import get_model
from src.load_data import Loader
from src.train_model import Trainer
from src.evaluate_model import Evaluator


def cli_main(flags):
    Model = get_model(flags.class_type)

    model = Model(name=flags.name, save_dir=flags.save_dir, lr=flags.learning_rate, batch_size=flags.batch_size)

    loader = Loader(location=flags.data_dir, batch_size=flags.batch_size)

    trainer = Trainer(model)
    trainer.train(loader=loader, epochs=flags.epochs)

    evaluator = Evaluator(model)
    evaluator.evalute(classes=loader.classes, loader=loader.train_loader, type="train")
    evaluator.evalute(classes=loader.classes, loader=loader.test_loader, type="test")


if __name__ == "__main__":

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()

        parser.add_argument('--epochs', type=int,
                            default=1,
                            help='The number of epochs for training')

        parser.add_argument('--learning-rate', type=float,
                            default=0.001,
                            help='The learning rate to use during training')

        parser.add_argument('--class-type', type=str,
                            default="example",
                            help='The class of network to use')

        parser.add_argument('--name', type=str,
                            default="example",
                            help='The name to save results as')

        parser.add_argument('--save-dir', type=str,
                            default="./results",
                            help='The base directory to store results in')

        parser.add_argument('--data-dir', type=str,
                            default="./data",
                            help='The location of the data on the disk')

        parser.add_argument('--batch-size', type=int,
                            default=400,
                            help='The batch size to use for feeding examples')

        parsed_flags, _ = parser.parse_known_args()

        cli_main(parsed_flags)
