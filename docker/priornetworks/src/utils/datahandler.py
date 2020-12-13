import os
import sys

from torchvision import datasets, transforms

from utils.standardised_datasets import TinyImageNet
from utils.utils import random_subset

NORMALIZATION_PARAMETERS = {'CIFAR10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2112, 0.2086, 0.2121]},
                            'SVHN': {'mean': [0.4377, 0.4438, 0.4728], 'std': [0.1287, 0.1320, 0.1128]},
                            'TIM': {'mean': [0.4802, 0.4481, 0.3975], 'std': [0.2382, 0.2342, 0.2356]}
                            # from train only
                            }


class DataHandler:
    def __init__(self, id_dataset, ood_dataset, data_dir="resources/data", individual_normalization=False,
                 training_sample_size=None, test_sample_size=None):

        os.makedirs(data_dir, exist_ok=True)

        IMAGE_SIZE = 32

        id_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(NORMALIZATION_PARAMETERS[id_dataset]['mean'],
                                                                NORMALIZATION_PARAMETERS[id_dataset]['std']),
                                           transforms.Resize(IMAGE_SIZE),
                                           transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE))])

        if individual_normalization:
            ood_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(NORMALIZATION_PARAMETERS[ood_dataset]['mean'],
                                                                     NORMALIZATION_PARAMETERS[ood_dataset]['std']),
                                                transforms.Resize(IMAGE_SIZE),
                                                transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE))])
        else:
            ood_transform = id_transform

        if id_dataset == "CIFAR10":
            id_train_dataset = datasets.CIFAR10(root=f"{data_dir}/CIFAR10", download=True, transform=id_transform,
                                                target_transform=None, train=True)
            id_test_dataset = datasets.CIFAR10(root=f"{data_dir}/CIFAR10", download=True, transform=id_transform,
                                               target_transform=None, train=False)
        else:
            sys.exit(f"{id_dataset} is not a valid name for an id_dataset. Options are: 'CIFAR10'")

        if ood_dataset == "SVHN":
            ood_train_dataset = datasets.SVHN(root=f"{data_dir}/SVHN", download=True, transform=ood_transform,
                                              target_transform=None, split='train')
            ood_test_dataset = datasets.SVHN(root=f"{data_dir}/SVHN", download=True, transform=ood_transform,
                                             target_transform=None, split='test')
        elif ood_dataset == "TIM":
            ood_train_dataset = TinyImageNet(root=f"{data_dir}/TIM", transform=ood_transform, target_transform=None,
                                             split="train")
            ood_test_dataset = TinyImageNet(root=f"{data_dir}/TIM", transform=ood_transform, target_transform=None,
                                            split="test")
        elif ood_dataset == "Random":
            ood_train_dataset = datasets.FakeData(size=len(id_train_dataset), image_size=(3, IMAGE_SIZE, IMAGE_SIZE),
                                                  num_classes=10, transform=ood_transform, target_transform=None)
            ood_test_dataset = datasets.FakeData(size=len(id_test_dataset), image_size=(3, IMAGE_SIZE, IMAGE_SIZE),
                                                 num_classes=10, transform=ood_transform, target_transform=None)
        else:
            sys.exit(f"{ood_dataset} is not a valid name for an ood_dataset. Options are: 'SVHN', 'TIM'")

        self.training_sample_size = training_sample_size or min(len(id_train_dataset), len(ood_train_dataset))
        self.test_sample_size = test_sample_size or min(len(id_test_dataset), len(ood_test_dataset))

        self.id_train_dataset = random_subset(id_train_dataset, self.training_sample_size)
        self.ood_train_dataset = random_subset(ood_train_dataset, self.training_sample_size)
        self.id_test_dataset = random_subset(id_test_dataset, self.test_sample_size)
        self.ood_test_dataset = random_subset(ood_test_dataset, self.test_sample_size)
