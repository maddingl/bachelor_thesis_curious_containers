import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# import lmdb


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
    # std = torch.zeros(n_channels)
    # for batch, _ in loader:
    #     # Rearrange batch to be the shape of [B, C, W * H]
    #     batch = batch.view(batch.size(0), batch.size(1), -1)
    #     # Update total number of images
    #     nimages += batch.size(0)
    #     # Compute mean and std here
    #     mean += batch.mean(2).sum(0)
    #     std += batch.std(2).sum(0)
    #
    # # Final step
    # mean /= nimages
    # std /= nimages
    for i_batch, batch_target in enumerate(loader):
        batch = batch_target[0]
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0)
        var += batch.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    return mean, std


# (tensor([0.4914, 0.4822, 0.4465]), tensor([0.2023, 0.1994, 0.2010]))

if __name__ == '__main__':
    DATA_DIR = "resources/data"

    # dataset = datasets.MNIST(root=DATA_DIR,
    #                          download=True,
    #                          transform=transforms.Compose([transforms.ToTensor()]),
    #                          target_transform=None,
    #                          train=True)
    # dataset = datasets.SVHN(root=DATA_DIR,
    #                         download=True,
    #                         transform=transforms.Compose([transforms.ToTensor()]),
    #                         target_transform=None,
    #                         split='train')
    dataset = datasets.LSUN(root=DATA_DIR,
                            transform=transforms.Compose([transforms.ToTensor()]),
                            target_transform=None,
                            classes='train')

    # dataset = datasets.Omniglot(root=DATA_DIR,
    #                             download=True,
    #                             transform=transforms.Compose([transforms.ToTensor()]),
    #                             target_transform=None,
    #                             background=True)

    print(calc_mean_std(dataset, 1, 1000))

    # TODO:
    # STD für CIFAR10 hier anders als in main_train_new_network.py -> warum?
    # überlegen, wie man hier Ergebnisse erzielen und gut zwischenspeichern kann
    # Bewusst SVHN-Daten anders normalisieren und gleicher Normalisierung wie CIFAR10 gegenüberstellen.
    # Überlegung: Real World --> wie werden Daten hier normalisiert? Es kann ja nicht unterschieden werden, ob ID oder OOD --> Mit Voigt drüber sprechen.
