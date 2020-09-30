#!/usr/bin/python3


class Hyperparams:
    def __init__(self,
                 target_concentration=100,
                 concentration=1,
                 reverse_kld=True,
                 lr=7.5e-4,
                 optimizer='sgd',
                 momentum=0.9,
                 weight_decay=0,
                 batch_size=128,
                 clip_norm=10):
        self.target_concentration = target_concentration or 100
        self.concentration = concentration or 1
        self.reverse_kld = reverse_kld or True
        self.lr = lr or 7.5e-4
        self.optimizer = optimizer or 'sgd'
        self.momentum = momentum or 0.9
        self.weight_decay = weight_decay or 0
        self.batch_size = batch_size or 128
        self.clip_norm = clip_norm or 10
