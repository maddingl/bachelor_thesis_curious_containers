#!/usr/bin/python3

import argparse
import csv
import os
import time

import torch
from torchvision import models, datasets, transforms

from utils.training import Trainer
from utils.hyperparams import Hyperparams
from utils.utils import random_subset

parser = argparse.ArgumentParser(description='Train a Dirichlet Prior Network model using a '
                                             'standard Torchvision architecture on a Torchvision '
                                             'dataset.')

parser.add_argument('--name', type=str, required=True, help='Name of the experiment')

parser.add_argument('--training_sample_size', type=int, required=False,
                    help='Amount of samples to be used for training in each in-domain and out-of-domain-dataset. '
                         'If not set, min(len(id_train_dataset), len(ood_train_dataset)) is used')

parser.add_argument('--test_sample_size', type=int, required=False,
                    help='Amount of samples to be used for testing in each in-domain and out-of-domain-dataset. '
                         'If not set, min(len(id_test_dataset), len(ood_test_dataset)) is used')

parser.add_argument('--n_epochs', type=int, default=20,
                    help='Amount of epochs to be run. Default is 20.')

parser.add_argument('--target_concentration', type=float, default=100,
                    help='Hyperparameter: target concentration. Default is 100.')
parser.add_argument('--concentration', type=float, default=1,
                    help='Hyperparameter: concentration. Default is 1.')
parser.add_argument('--reverse_kld', type=bool, default=True,
                    help='Hyperparameter: whether or not to reverse the Kullback-Leibler-Divergence. Default is True.')
parser.add_argument('--lr', type=float, default=7.5e-4,
                    help='Hyperparameter: learning rate. Default is 7.5e-4.')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='Hyperparameter: optimizer: \'sgd\' or \'adamW\'. Default is \'sgd\'.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Hyperparameter: momentum. Default is 0.9.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Hyperparameter: weight decay. Default is 0.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Hyperparameter: batch size. Default is 128.')
parser.add_argument('--clip_norm', type=float, default=10,
                    help='Hyperparameter: clip norm for gradient clipping. Default is 10.')

parser.add_argument('--individual_normalization', type=bool, default=False,
                    help='Boolean: whether to normalize id and ood individually or all same as id.')

if __name__ == '__main__':
    args = parser.parse_args()

    # print name
    print(args.name)

    # setup model
    MODEL_DIR = "resources/model"
    os.makedirs(MODEL_DIR, exist_ok=True)

    NUM_CLASSES = 10

    model = models.vgg16(num_classes=NUM_CLASSES)
    torch.save(model.state_dict(), f"{MODEL_DIR}/{args.name}_inital_model_{time.strftime('%d-%m-%Y_%H-%M-%S')}.tar")

    # setup datasets

    DATA_DIR = "resources/data"
    os.makedirs(DATA_DIR, exist_ok=True)

    NORMALIZATION_PARAMETERS = {'MNIST': {'mean': 0.1307, 'std': 0.3052},
                                'Omniglot': {'mean': 0.9221, 'std': 0.2666},
                                'CIFAR10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2112, 0.2086, 0.2121]},
                                'SVHN': {'mean': [0.4377, 0.4438, 0.4728], 'std': [0.1287, 0.1320, 0.1128]}}

    # DATASET_DICT = {'MNIST': datasets.MNIST,
    #                 'SVHN': datasets.SVHN,
    #                 'CIFAR10': datasets.CIFAR10,
    #                 'CIFAR100': datasets.CIFAR100,
    #                 'LSUN': datasets.LSUN
    #                 }

    # mean_cifar10 = (0.4914, 0.4823, 0.4465)  # this is the CIFAR10-mean.
    # std_cifar10 = (0.247, 0.243, 0.261)  # this is the CIFAR10-std.
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_cifar10, std_cifar10)])

    id_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(NORMALIZATION_PARAMETERS['CIFAR10']['mean'],
                                                            NORMALIZATION_PARAMETERS['CIFAR10']['std'])])

    if args.individual_normalization:
        ood_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(NORMALIZATION_PARAMETERS['SVHN']['mean'],
                                                                 NORMALIZATION_PARAMETERS['SVHN']['std'])])
    else:
        ood_transform = id_transform

    # id_train_dataset = datasets.MNIST(root=DATA_DIR,
    #                                   download=True,
    #                                   # transform=transform,
    #                                   target_transform=None,
    #                                   train=True)

    id_train_dataset = datasets.CIFAR10(root=DATA_DIR,
                                        download=True,
                                        transform=id_transform,
                                        target_transform=None,
                                        train=True)

    id_test_dataset = datasets.CIFAR10(root=DATA_DIR,
                                       download=True,
                                       transform=id_transform,
                                       target_transform=None,
                                       train=False)

    ood_train_dataset = datasets.SVHN(root=DATA_DIR,
                                      download=True,
                                      transform=ood_transform,
                                      target_transform=None,
                                      split='train')

    ood_test_dataset = datasets.SVHN(root=DATA_DIR,
                                     download=True,
                                     transform=ood_transform,
                                     target_transform=None,
                                     split='test')

    training_sample_size = args.training_sample_size or min(len(id_train_dataset), len(ood_train_dataset))
    test_sample_size = args.test_sample_size or min(len(id_test_dataset), len(ood_test_dataset))

    print(f"training sample size: {training_sample_size}")
    print(f"test sample size: {test_sample_size}")

    id_train_dataset = random_subset(id_train_dataset, training_sample_size)
    ood_train_dataset = random_subset(ood_train_dataset, training_sample_size)
    id_test_dataset = random_subset(id_test_dataset, test_sample_size)
    ood_test_dataset = random_subset(ood_test_dataset, test_sample_size)

    hyperparams = Hyperparams(args.target_concentration, args.concentration, args.reverse_kld, args.lr, args.optimizer,
                              args.momentum, args.weight_decay, args.batch_size, args.clip_norm)

    n_epochs = args.n_epochs

    # Setup model trainer and train model
    trainer = Trainer(model=model,
                      ood_dataset=ood_train_dataset,
                      test_ood_dataset=ood_test_dataset,
                      train_dataset=id_train_dataset,
                      test_dataset=id_test_dataset,
                      hyperparams=hyperparams)
    trainer.train(n_epochs)

    # Save final model
    torch.save(model.state_dict(), f"{MODEL_DIR}/{args.name}_trained_model_{time.strftime('%d-%m-%Y_%H-%M-%S')}.tar")
