from utils.datahandler import DataHandler
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from utils.graphics import Graphics
from utils.normalization_parameter_calculator import calc_mean_std
from utils.standardised_datasets import TinyImageNet
from torchvision import datasets, transforms

if __name__ == '__main__':
    # graphics = Graphics("test")
    # graphics.plot_loss()
    # graphics.plot_accuracy()

    # data_dir = "resources/data"
    # dataset = TinyImageNet(root=f"{data_dir}/TIM",
    #                                  transform=transforms.Compose([transforms.ToTensor()]),
    #                                  target_transform=None,
    #                                  split="train")

    data_handler = DataHandler('CIFAR10',
                               'Random')

    w, h = 32, 32
    # data = np.zeros((h, w, 3), dtype=np.uint8)
    # data[0:256, 0:256] = [255, 0, 0]  # red patch in upper left
    print(data_handler.ood_train_dataset[0][0].size())
    plt.imshow(data_handler.ood_train_dataset[0][0].permute(1, 2, 0))
    plt.show()
    # img = Image.fromarray(data_handler.ood_train_dataset[0][0], 'RGB')
    # img.save('my.png')
    # img.show()
