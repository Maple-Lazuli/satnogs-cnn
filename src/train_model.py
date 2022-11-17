import datetime
import json
import os

import numpy as np
import torch

from src.load_data import Loader
from models.example import Model


def save_training_performance(model, train_loss, val_loss, all_step_loss):
    save_dir = model.save_dir
    with open(os.path.join(save_dir, "training.json"), "w") as file_out:
        json.dump({
            "name": model.name,
            "time": str(datetime.datetime.now()),
            "lr": model.lr,
            "batch_size": model.batch_size,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "all_step_loss": all_step_loss
        }, file_out)


class Trainer:
    def __init__(self, model):
        self.model = model

    def train(self, loader, epochs):
        train_loader = loader.train_loader
        val_loader = loader.test_loader

        train_loss = []
        val_loss = []
        all_step_loss = []
        for epoch in range(epochs):
            loss = []
            for batch_idx, data_target in enumerate(train_loader):
                data = data_target[0]
                target = data_target[1]
                loss.append(self.model.step(data, target))

            all_step_loss.append(loss)
            mean_step_loss = np.mean(loss)
            print(f"Epoch {epoch+1} loss {mean_step_loss}")
            train_loss.append(mean_step_loss)

            with torch.no_grad():
                loss = []
                for data_target in val_loader:
                    data = data_target[0]
                    target = data_target[1]
                    loss.append(self.model.loss(data, target))
            mean_step_loss = np.mean(loss)
            val_loss.append(mean_step_loss)

        save_training_performance(self.model, train_loss, val_loss, all_step_loss)

        self.model.save()


if __name__ == "__main__":

    model = Model()
    loader = Loader(batch_size=model.batch_size)

    trainer = Trainer(model)
    trainer.train(loader=loader, epochs=3)
