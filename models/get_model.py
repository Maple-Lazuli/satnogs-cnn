import models.lenet as lenet
import models.babycnn as babycnn


def get_model(name):
    if name == "lenet":
        return lenet.Model
    elif name == "babycnn":
        return babycnn.Model

