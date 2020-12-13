#!/usr/bin/python3
import os
import sys
import time

import torch
from torchvision import models


class ModelHandler:
    def __init__(self, model_dir="resources/model", num_classes=10, arch="vgg16"):
        self.model_dir = model_dir
        if arch == "vgg16":
            self.model = models.vgg16(num_classes=num_classes)
        else:
            sys.exit(f"{arch} is not a valid model architecture! Options: 'vgg16'")

    def save(self, name):
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.model_dir}/{name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}.tar")

    def load(self, name):
        self.model.load_state_dict(torch.load(f"{self.model_dir}/{name}"))
