#!/usr/bin/python3

import os
import sys

import torch
from torchvision import models, datasets, transforms
from utils.utils import random_subset

NORMALIZATION_PARAMETERS = {'MNIST': {'mean': 0.1307, 'std': 0.3052},
                            'Omniglot': {'mean': 0.9221, 'std': 0.2666},
                            'CIFAR10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2112, 0.2086, 0.2121]},
                            'SVHN': {'mean': [0.4377, 0.4438, 0.4728], 'std': [0.1287, 0.1320, 0.1128]}}


class DataHandler:
    def __init__(self, id_dataset, ood_dataset, data_dir="resources/data", individual_normalization=False,
                 training_sample_size=None, test_sample_size=None):

        os.makedirs(data_dir, exist_ok=True)

        # DATASET_DICT = {'MNIST': datasets.MNIST,
        #                 'SVHN': datasets.SVHN,
        #                 'CIFAR10': datasets.CIFAR10,
        #                 'CIFAR100': datasets.CIFAR100,
        #                 'LSUN': datasets.LSUN
        #                 }

        # mean_cifar10 = (0.4914, 0.4823, 0.4465)  # this is the CIFAR10-mean.
        # std_cifar10 = (0.247, 0.243, 0.261)  # this is the CIFAR10-std.
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_cifar10, std_cifar10)])
        IMAGE_SIZE = 32 # TODO: Maybe differentiate between width and height

        id_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(NORMALIZATION_PARAMETERS[id_dataset]['mean'],
                                                                NORMALIZATION_PARAMETERS[id_dataset]['std']),
                                           transforms.Resize(IMAGE_SIZE),
                                           transforms.CenterCrop((IMAGE_SIZE,IMAGE_SIZE))])

        if individual_normalization:
            ood_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(NORMALIZATION_PARAMETERS[ood_dataset]['mean'],
                                                                     NORMALIZATION_PARAMETERS[ood_dataset]['std']),
                                           transforms.Resize(IMAGE_SIZE),
                                           transforms.CenterCrop((IMAGE_SIZE,IMAGE_SIZE))])
        else:
            ood_transform = id_transform

        # id_train_dataset = datasets.MNIST(root=DATA_DIR,
        #                                   download=True,
        #                                   # transform=transform,
        #                                   target_transform=None,
        #                                   train=True)
        if id_dataset == "CIFAR10":
            id_train_dataset = datasets.CIFAR10(root=data_dir, download=True, transform=id_transform,
                                                target_transform=None, train=True)
            id_test_dataset = datasets.CIFAR10(root=data_dir, download=True, transform=id_transform,
                                               target_transform=None, train=False)
        # TODO: put in elifs for other datasets
        else:
            sys.exit(f"{id_dataset} is not a valid name for an id_dataset. Options are: 'CIFAR10'")

        if ood_dataset == "SVHN":
            ood_train_dataset = datasets.SVHN(root=data_dir, download=True, transform=ood_transform,
                                              target_transform=None, split='train')
            ood_test_dataset = datasets.SVHN(root=data_dir, download=True, transform=ood_transform,
                                             target_transform=None, split='test')
        elif ood_dataset == "CelebA":
            ood_train_dataset = datasets.CelebA(root=data_dir, download=True, transform=ood_transform,
                                                target_transform=None, split='train')
            ood_test_dataset = datasets.CelebA(root=data_dir, download=True, transform=ood_transform,
                                                target_transform=None, split='test')
        # TODO: put in elifs for other datasets
        else:
            sys.exit(f"{ood_dataset} is not a valid name for an ood_dataset. Options are: 'SVHN', 'CelebA'")

        self.training_sample_size = training_sample_size or min(len(id_train_dataset), len(ood_train_dataset))
        self.test_sample_size = test_sample_size or min(len(id_test_dataset), len(ood_test_dataset))

        self.id_train_dataset = random_subset(id_train_dataset, self.training_sample_size)
        self.ood_train_dataset = random_subset(ood_train_dataset, self.training_sample_size)
        self.id_test_dataset = random_subset(id_test_dataset, self.test_sample_size)
        self.ood_test_dataset = random_subset(ood_test_dataset, self.test_sample_size)
