import os

from .training import Trainer
import matplotlib.pyplot as plt

SAVE_DIR = "resources/graphics"
os.makedirs(SAVE_DIR, exist_ok=True)


class Graphics:
    def __init__(self, name: str, trainer: Trainer = None):
        self.name = name
        self.trainer = trainer

    def plot_loss(self):
        plt.clf()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        if self.trainer is not None:
            plt.plot(range(1, len(self.trainer.test_loss + 1)), self.trainer.test_loss)
        else:
            plt.plot(range(1, 7), [1, 4, 6, 7.5, 1, 4])
        plt.savefig(f"{SAVE_DIR}/{self.name}_loss")

    def plot_accuracy(self):
        plt.clf()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        if self.trainer is not None:
            plt.plot(range(1, len(self.trainer.test_accuracy + 1)), self.trainer.test_accuracy)
        else:
            plt.plot(range(1, 7), [1, 4, 1, 7.5, 1, 4])
        plt.savefig(f"{SAVE_DIR}/{self.name}_accuracy")
