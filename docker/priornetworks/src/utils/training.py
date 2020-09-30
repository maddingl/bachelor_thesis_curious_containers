#!/usr/bin/python3

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from utils.hyperparams import Hyperparams
from utils.loss import DirichletKLLoss, PriorNetMixedLoss
from utils.utils import dirichlet_prior_network_uncertainty, calc_accuracy_torch


class Trainer:
    def __init__(self, model,
                 train_dataset,
                 ood_dataset,
                 test_dataset,
                 test_ood_dataset,
                 hyperparams,
                 log_interval=100):

        # Die Klasse benötigt ein Modell und vier Datasets als Input:
        # Für In-Domain und für Out-Of-Domain je ein Trainings- und ein Test-Dataset.
        #
        # Außerdem werden die Hyperparameter benötigt.
        # Das log_interval bestimmt, wie oft Ergebnisse zwischendurch geloggt werden. # TODO: check, ob das aktuell sinnvoll ist mit 100

        assert isinstance(model, nn.Module)
        assert isinstance(train_dataset, Dataset)
        assert isinstance(ood_dataset, Dataset)
        assert isinstance(test_dataset, Dataset)
        assert isinstance(test_ood_dataset, Dataset)
        assert isinstance(hyperparams, Hyperparams)

        self.model = model
        self.hyperparams = hyperparams
        self.log_interval = log_interval

        id_criterion = DirichletKLLoss(target_concentration=hyperparams.target_concentration,
                                       concentration=hyperparams.concentration,
                                       reverse=hyperparams.reverse_kld)

        ood_criterion = DirichletKLLoss(target_concentration=0.0,
                                        concentration=hyperparams.concentration,
                                        reverse=hyperparams.reverse_kld)
        criterion = PriorNetMixedLoss(id_criterion, ood_criterion)

        # Zum Trainieren und zum Testen wird dieselbe Loss-Funktion genutzt,
        # die durch die Instanz der PriorNetMixedLoss-Klasse repräsentiert wird.

        self.criterion = criterion
        self.test_criterion = criterion

        # Der Optimizer wird mit den dafür benötigten Hyperparametern initialisiert.
        assert (hyperparams.optimizer in ['sgd', 'adamW'])
        if hyperparams.optimizer == 'sgd':
            optimizer = optim.SGD
            optimizer_params = {'lr': hyperparams.lr,
                                'momentum': hyperparams.momentum,
                                'nesterov': True,
                                'weight_decay': hyperparams.weight_decay}
        else:
            optimizer = optim.AdamW
            optimizer_params = {'lr': hyperparams.lr,
                                'betas': (hyperparams.momentum, 0.999),
                                'weight_decay': hyperparams.weight_decay}
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)

        # Für den korrekten Ablauf sind folgende Bedingungen wichtig:
        assert len(train_dataset) == len(ood_dataset)
        assert len(test_dataset) == len(test_ood_dataset)

        # Diese DataLoader werden später zum Laden der Batches benötigt.
        # Beim Trainieren sollten die Daten randomisiert werden, beim Testen ist das nicht nötig.
        self.trainloader = DataLoader(train_dataset,
                                      batch_size=hyperparams.batch_size,
                                      shuffle=True)
        self.testloader = DataLoader(test_dataset,
                                     batch_size=hyperparams.batch_size,
                                     shuffle=False)
        self.oodloader = DataLoader(ood_dataset, batch_size=hyperparams.batch_size,
                                    shuffle=True, num_workers=1)
        self.test_oodloader = DataLoader(test_ood_dataset, batch_size=hyperparams.batch_size,
                                         shuffle=False, num_workers=1)

        # Lists for storing training metrics
        self.train_loss, self.train_accuracy, self.train_eval_steps = [], [], []
        # Lists for storing test metrics
        self.test_loss, self.test_accuracy, self.test_eval_steps = [], [], []
        self.steps: int = 0

    def train(self, n_epochs):
        assert isinstance(n_epochs, int)

        for epoch in range(n_epochs):
            print(f'Training epoch: {epoch + 1} / {n_epochs}')
            # Train
            start = time.time()
            self._train_single_epoch()
            # Test
            self.test(time=time.time() - start)
        return

    def _train_single_epoch(self):
        # Set model in train mode
        self.model.train()

        # Initialisiere Auswertungsvariablen
        accuracies = 0.0
        train_loss = 0.0
        id_alpha_0, ood_alpha_0 = 0.0, 0.0

        # Es folgt ein typischer PyTorch-Trainingsprozess
        for i, (data, ood_data) in enumerate(zip(self.trainloader, self.oodloader)):
            # Get inputs
            inputs, labels = data
            ood_inputs, _ = ood_data

            # zero the parameter gradients
            self.optimizer.zero_grad()

            inputs = torch.cat((inputs, ood_inputs), dim=0)  # konkateniere ID-inputs und OOD-inputs
            outputs = self.model(inputs)  # forward-pass durch das Modell
            id_outputs, ood_outputs = torch.chunk(outputs, 2, dim=0)  # teile Outputs wieder in ID und OOD
            loss = self.criterion(id_outputs, ood_outputs,
                                  labels)  # Loss-Berechnung, criterion ist ein PriorNetMixedLoss
            assert torch.all(torch.isfinite(loss)).item()
            loss.backward()  # Berechnung der Gradienten durch Backpropagation
            clip_grad_norm_(self.model.parameters(),
                            self.hyperparams.clip_norm)  # prevent gradients from getting too large
            self.optimizer.step()  # Anpassung der Modellparameter

            # Update the number of steps
            self.steps += 1

            # log statistics
            id_alpha_0 += torch.mean(torch.sum(torch.exp(id_outputs), dim=1)).item()
            ood_alpha_0 += torch.mean(torch.sum(torch.exp(ood_outputs), dim=1)).item()

            probs = F.softmax(id_outputs, dim=1)
            accuracy = calc_accuracy_torch(probs, labels).item()
            accuracies += accuracy
            train_loss += loss.item()
            if self.steps % self.log_interval == 0:
                self.train_accuracy.append(accuracy)
                self.train_loss.append(loss.item())
                self.train_eval_steps.append(self.steps)

        # zuvor wurde für jeden Batch das Ergebnis addiert,
        # hier muss es noch durch die Anzahl der Batches geteilt werden, damit ein Durchschnitt entsteht.
        accuracies /= len(self.trainloader)
        train_loss /= len(self.trainloader)
        id_alpha_0 /= len(self.trainloader)
        ood_alpha_0 /= len(self.trainloader)

        # Auswertung nach stdout und in eine LOG.txt

        print(f"Train Loss: {np.round(train_loss, 3)}; "
              f"Train Error: {np.round(100.0 * (1.0 - accuracies), 1)}; "
              f"Train ID precision: {np.round(id_alpha_0, 1)}; "
              f"Train OOD precision: {np.round(ood_alpha_0, 1)}")

        with open('./LOG.txt', 'a') as f:
            f.write(f"Train Loss: {np.round(train_loss, 3)}; "
                    f"Train Error: {np.round(100.0 * (1.0 - accuracies), 1)}; "
                    f"Train ID precision: {np.round(id_alpha_0, 1)}; "
                    f"Train OOD precision: {np.round(ood_alpha_0, 1)}; ")
        return

    def test(self, time):
        """
        Single evaluation on the entire provided test dataset.
        Return accuracy, mean test loss, and an array of predicted probabilities
        """
        test_loss, accuracy = 0.0, 0.0

        id_logits, ood_logits = [], []

        # Set model in eval mode
        self.model.eval()
        id_alpha_0, ood_alpha_0 = 0.0, 0.0
        with torch.no_grad():
            for i, (data, ood_data) in enumerate(zip(self.testloader, self.test_oodloader)):
                # Get inputs
                id_inputs, labels = data
                ood_inputs, _ = ood_data
                assert torch.isnan(id_inputs).sum() == 0
                assert torch.isnan(ood_inputs).sum() == 0
                id_outputs = self.model(id_inputs)
                ood_outputs = self.model(ood_inputs)
                probs = F.softmax(id_outputs, dim=1)

                accuracy += calc_accuracy_torch(probs, labels).item()
                test_loss += self.test_criterion(id_outputs, ood_outputs, labels).item()

                # Get in-domain and OOD Precision
                id_alpha_0 += torch.mean(torch.sum(torch.exp(id_outputs), dim=1)).item()
                ood_alpha_0 += torch.mean(torch.sum(torch.exp(ood_outputs), dim=1)).item()

                # Append logits for future OOD detection at test time calculation...
                id_logits.append(id_outputs.numpy())
                ood_logits.append(ood_outputs.numpy())

        id_alpha_0 = id_alpha_0 / len(self.testloader)
        ood_alpha_0 = ood_alpha_0 / len(self.test_oodloader)

        test_loss = test_loss / len(self.testloader)
        accuracy = accuracy / len(self.testloader)

        id_logits = np.concatenate(id_logits, axis=0)
        ood_logits = np.concatenate(ood_logits, axis=0)
        logits = np.concatenate([id_logits, ood_logits], axis=0)

        # Berechnung der Unsicherheit aus den logits. Hier wird 'differential_entropy' als Maß gewählt,
        # da mit dieser Art der Auswertung im Paper die besten AUROC-Werte erzielt wurden.
        uncertainties = dirichlet_prior_network_uncertainty(logits)

        in_domain = np.zeros(shape=[id_logits.shape[0]], dtype=np.int32)
        ood_domain = np.ones(shape=[ood_logits.shape[0]], dtype=np.int32)
        domain_labels = np.concatenate([in_domain, ood_domain], axis=0)
        # domain_labels ist ein Vektor, der für jedes in-domain-sample eine 0
        # und für jedes out-of-domain-sample ein 1 enthält.

        auc_MI = roc_auc_score(domain_labels, uncertainties['mutual_information'])
        auc_DE = roc_auc_score(domain_labels, uncertainties['differential_entropy'])

        print(f"Test Loss: {np.round(test_loss, 3)}; "
              f"Test Error: {np.round(100.0 * (1.0 - accuracy), 1)}%; "
              f"Test ID precision: {np.round(id_alpha_0, 1)}; "
              f"Test OOD precision: {np.round(ood_alpha_0, 1)}; "
              f"Test AUROC (MI): {np.round(100.0 * auc_MI, 1)}; "
              f"Test AUROC (DE): {np.round(100.0 * auc_DE, 1)}; "
              f"Time Per Epoch: {np.round(time / 60.0, 1)} min")

        # Log statistics
        self.test_loss.append(test_loss)
        self.test_accuracy.append(accuracy)
        self.test_eval_steps.append(self.steps)
        return
