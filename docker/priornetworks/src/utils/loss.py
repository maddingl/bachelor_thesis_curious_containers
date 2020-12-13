from typing import Optional

import torch


def dirichlet_kl_divergence(alphas, target_alphas, precision=None, target_precision=None,
                            epsilon=1e-8):  # see supplementary C5
    """
    This function computes the Forward KL divergence between a model Dirichlet distribution
    and a target Dirichlet distribution based on the concentration (alpha) parameters of each.

    :param alphas: Tensor containing concentration parameters of model. Expected shape is batchsize X num_classes.
    :param target_alphas: Tensor containing target concentration parameters. Expected shape is batchsize X num_classes.
    :param precision: Optional argument. Can pass in precision of model. Expected shape is batchsize X 1
    :param target_precision: Optional argument. Can pass in target precision. Expected shape is batchsize X 1
    :param epsilon: Smoothing factor for numerical stability. Default value is 1e-8
    :return: Tensor for batchsize X 1 of forward KL divergences between target Dirichlet and model
    """
    if not precision:
        precision = torch.sum(alphas, dim=1, keepdim=True)
    if not target_precision:
        target_precision = torch.sum(target_alphas, dim=1, keepdim=True)
    precision_term = torch.lgamma(target_precision) - torch.lgamma(precision)
    alphas_term = torch.sum(torch.lgamma(alphas + epsilon) - torch.lgamma(target_alphas + epsilon)
                            + (target_alphas - alphas) * (torch.digamma(target_alphas + epsilon)
                                                          - torch.digamma(target_precision + epsilon))
                            , dim=1, keepdim=True)

    cost = torch.squeeze(precision_term + alphas_term)
    return cost


class DirichletKLLoss:
    """
    Can be applied to any model which returns logits
    """

    def __init__(self, target_concentration=1e3, concentration=1.0, reverse=True):
        """
        :param target_concentration: The concentration parameter for the
        target class (if provided)
        :param concentration: The 'base' concentration parameters for
        non-target classes.
        """
        self.target_concentration = torch.tensor(target_concentration,
                                                 dtype=torch.float32)
        self.concentration = concentration
        self.reverse = reverse

    def __call__(self, logits, labels):
        alphas = torch.exp(logits)
        return self.forward(alphas, labels)

    def forward(self, alphas, labels):
        loss = self.compute_loss(alphas, labels)
        return torch.mean(loss)

    def compute_loss(self, alphas, labels: Optional[torch.tensor] = None):
        """
        :param alphas: The alpha parameter outputs from the model
        :param labels: Optional. The target labels indicating the correct
        class.

        The loss creates a set of target alpha (concentration) parameters
        with all values set to self.concentration, except for the correct
        class (if provided), which is set to self.target_concentration
        :return: an array of per example loss
        """
        target_alphas = torch.ones_like(alphas) * self.concentration
        if labels is not None:
            target_alphas += \
                torch.zeros_like(alphas) \
                    .scatter_(1, labels[:, None], self.target_concentration.repeat(alphas.shape[0])[:, None])  # !!

        if self.reverse:
            loss = dirichlet_kl_divergence(alphas=target_alphas, target_alphas=alphas)
        else:
            loss = dirichlet_kl_divergence(alphas=alphas, target_alphas=target_alphas)
        return loss


class PriorNetMixedLoss:
    def __init__(self, id_loss, ood_loss):
        self.id_loss = id_loss
        self.ood_loss = ood_loss

    def __call__(self, id_outputs, ood_outputs, labels):
        return self.forward(id_outputs, ood_outputs, labels)

    def forward(self, id_outputs, ood_outputs, labels):
        return self.id_loss(id_outputs, labels) + self.ood_loss(ood_outputs, None)
