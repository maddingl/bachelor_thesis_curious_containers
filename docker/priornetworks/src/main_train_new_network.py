import argparse
import os

from utils.datahandler import DataHandler
from utils.modelhandler import ModelHandler
from utils.training import Trainer
from utils.hyperparams import Hyperparams

parser = argparse.ArgumentParser(description='Train a Dirichlet Prior Network model using a '
                                             'standard Torchvision architecture on a Torchvision '
                                             'dataset.')

parser.add_argument('--name', type=str, required=True, help='Name of the experiment')

parser.add_argument('--id_dataset', type=str, default='CIFAR10', help='in-domain dataset')

parser.add_argument('--ood_dataset', type=str, default='SVHN', help='out-of-domain dataset')

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

    print(args.name)

    model_handler = ModelHandler()
    model_handler.save(f"{args.name}_inital_model")

    data_handler = DataHandler(args.id_dataset,
                               args.ood_dataset,
                               training_sample_size=args.training_sample_size,
                               test_sample_size=args.test_sample_size,
                               individual_normalization=args.individual_normalization)

    hyperparams = Hyperparams(args.target_concentration, args.concentration, args.reverse_kld, args.lr, args.optimizer,
                              args.momentum, args.weight_decay, args.batch_size, args.clip_norm)

    n_epochs = args.n_epochs

    trainer = Trainer(model=model_handler.model,
                      ood_dataset=data_handler.ood_train_dataset,
                      test_ood_dataset=data_handler.ood_test_dataset,
                      train_dataset=data_handler.id_train_dataset,
                      test_dataset=data_handler.id_test_dataset,
                      hyperparams=hyperparams)
    trainer.train(n_epochs)

    trainer.save_results(args.name)

    model_handler.save(f"{args.name}_trained_model")
