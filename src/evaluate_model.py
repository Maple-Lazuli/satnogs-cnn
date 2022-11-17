"""
Reference: https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
"""

import datetime
import json
import os
import numpy as np
import torch

from data_utils.load_data import Loader
from models.babycnn import Model


def save_performance(matrix, classes, model, type):
    performance = {
        "name": model.name,
        "time": str(datetime.datetime.now()),
        "type": type,
        "Accuracy": [],
        "Classification Error": [],
        "TP": [],
        "FP": [],
        "TN": [],
        "FN": [],
        "Precision": [],
        "Recall": [],
        "Specificity": [],
        "F1-Score": [],
        "confusion_matrix": matrix.tolist(),
        "Classes": classes
    }

    for idx in range(len(classes)):
        TP = int(matrix[idx, idx])
        FN = int(np.sum(matrix[:, idx]) - TP)
        FP = int(np.sum(matrix[idx, :]) - TP)
        TN = int(np.sum(matrix) - (FN + FP + TP))
        performance["TP"].append(TP)
        performance["FN"].append(FN)
        performance["FP"].append(FP)
        performance["TN"].append(TN)
        if (TP + FP) != 0:
            performance['Precision'].append(TP / (TP + FP))
        else:
            performance['Precision'].append(0)

        if (TP + FN) != 0:
            performance['Recall'].append(TP / (TP + FN))
        else:
            performance['Recall'].append(0)

        if (TN + FP) != 0:
            performance['Specificity'].append(TN / (TN + FP))
        else:
            performance['Specificity'].append(0)

        if (TP + FP + TN + FN) != 0:
            performance['Accuracy'].append((TP + TN) / (TP + FP + TN + FN))
            performance['Classification Error'].append((FP + FN) / (TP + FP + TN + FN))
        else:
            performance['Accuracy'].append(0)
            performance['Classification Error'].append(0)

        if (performance['Precision'][idx] + performance['Recall'][idx]) != 0:
            performance['F1-Score'].append(((2 * performance['Precision'][idx] * performance['Recall'][idx]) /
                                                 (performance['Precision'][idx] + performance['Recall'][idx])))
        else:
            performance['F1-Score'].append(0)

    performance['Total Accuracy'] = ((np.sum(performance['TP']) + np.sum(performance['TN'])) /
                                     (np.sum(performance['TP'])
                                      + np.sum(performance['TN'])
                                      + np.sum(performance['FP'])
                                      + np.sum(performance['FN'])))

    with open(os.path.join(model.save_dir, f"{type}_evaluation.json"), "w") as file_out:
        json.dump(performance, file_out)

class Evaluator:
    def __init__(self, model):
        self.model = model

    def evalute(self, classes, loader, type):

        confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)

        with torch.no_grad():
            for data in loader:
                inputs, targets = data[0], data[1]

                outputs = self.model.classify(inputs)

                _, predicted = torch.max(outputs, 1)
                predicted = predicted.to("cpu")

                for pred, true in zip(predicted, targets):
                    confusion_matrix[int(pred), int(true)] += 1

        save_performance(matrix=confusion_matrix, classes=classes, model=self.model, type=type)


if __name__ == "__main__":
    model = Model()
    loader = Loader(batch_size=model.batch_size)
    model.load()
    evaluator = Evaluator(model)
    evaluator.evalute(classes=loader.classes, loader=loader.train_loader, type="train")
    evaluator.evalute(classes=loader.classes, loader=loader.test_loader, type="test")

