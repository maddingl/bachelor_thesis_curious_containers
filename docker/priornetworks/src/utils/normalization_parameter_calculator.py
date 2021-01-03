import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.tim_preparation import TinyImageNet


def calc_mean_std(dataset, n_channels, batch_size):
    # source: https://stackoverflow.com/questions/60101240/finding-mean-and-standard-deviation-across-image-channels-pytorch/60803379#60803379
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        shuffle=False
    )

    nimages = 0
    mean = torch.zeros(n_channels)
    var = torch.zeros(n_channels)
    for i_batch, batch_target in enumerate(loader):
        batch = batch_target[0]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        nimages += batch.size(0)
        mean += batch.mean(2).sum(0)
        var += batch.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    return mean, std


if __name__ == '__main__':
    data_dir = "resources/data"

    # dataset = datasets.CIFAR10(root=f"{data_dir}/CIFAR10",
    #                            download=True,
    #                            transform=transforms.Compose([transforms.ToTensor()]),
    #                            target_transform=None,
    #                            train=True)

    # dataset = datasets.SVHN(root=f"{data_dir}/SVHN",
    #                         download=True,
    #                         transform=transforms.Compose([transforms.ToTensor()]),
    #                         target_transform=None,
    #                         split='train')

    dataset = TinyImageNet(root=f"{data_dir}/TIM",
                                     transform=transforms.Compose([transforms.ToTensor()]),
                                     target_transform=None,
                                     split="train")

    print(calc_mean_std(dataset, 1, 1000))
