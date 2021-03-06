#!/usr/bin/python3

import argparse

from utils.datahandler import DataHandler
from utils.modelhandler import ModelHandler
from utils.training import Trainer
from utils.hyperparams import Hyperparams

parser = argparse.ArgumentParser(description='Test an existing Dirichlet Prior Network model.')

parser.add_argument('--name', type=str, required=True, help='Name of the experiment')

parser.add_argument('--model_name', type=str, required=True, help='Name of the model to test')

parser.add_argument('--id_dataset', type=str, default='CIFAR10', help='in-domain dataset')

parser.add_argument('--ood_dataset', type=str, default='SVHN', help='out-of-domain dataset')

parser.add_argument('--test_sample_size', type=int, required=False,
                    help='Amount of samples to be used for testing in each in-domain and out-of-domain-dataset. '
                         'If not set, min(len(id_test_dataset), len(ood_test_dataset)) is used')

parser.add_argument('--individual_normalization', type=bool, default=False,
                    help='Boolean: whether to normalize id and ood individually or all same as id.')

if __name__ == '__main__':
    args = parser.parse_args()

    print(args.name)

    model_handler = ModelHandler()
    model_handler.load(args.model_name)

    data_handler = DataHandler(args.id_dataset,
                               args.ood_dataset,
                               test_sample_size=args.test_sample_size,
                               individual_normalization=args.individual_normalization)

    hyperparams = Hyperparams()

    trainer = Trainer(model=model_handler.model,
                      ood_dataset=data_handler.ood_train_dataset,
                      test_ood_dataset=data_handler.ood_test_dataset,
                      train_dataset=data_handler.id_train_dataset,
                      test_dataset=data_handler.id_test_dataset,
                      hyperparams=hyperparams)

    trainer.test()
