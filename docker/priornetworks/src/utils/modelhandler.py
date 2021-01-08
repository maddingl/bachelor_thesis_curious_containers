import os
import sys
import time

import torch
from torchvision import models


class ModelHandler:
    """
    class for saving and loading models
    """
    def __init__(self, model_dir="resources/model", num_classes=10, arch="vgg16"):
        """
        initiates model in self.model
        @param model_dir: directory where to store and load models from
        @param num_classes: length of last layer
        @param arch: model architecture name
        """
        self.model_dir = model_dir
        if arch == "vgg16":
            self.model = models.vgg16(num_classes=num_classes)
        else:
            sys.exit(f"{arch} is not a valid model architecture! Options: 'vgg16'")

    def save(self, name):
        """
        saves self.model
        @param name: name of model
        """
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.model_dir}/{name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}.tar")

    def load(self, name):
        """
        loads a model into self.model
        @param name: full filename of model to load
        """
        self.model.load_state_dict(torch.load(f"{self.model_dir}/{name}"))
