import numpy as np
import torch
from scipy.special import gammaln, digamma
from torch.utils.data import random_split


def random_subset(dataset, length):
    """
    takes length random samples from dataset and returns a new dataset containing them
    @param dataset: dataset to take subset from
    @param length: amount of samples in subset
    @return: subset of dataset
    """
    assert (len(dataset) >= length)
    return random_split(dataset, [length, len(dataset) - length])[0]

def calc_accuracy_torch(y_probs, y_true):
    """
    @param y_probs: class probabilities matrix
    @param y_true: indices of target classes
    @return: accuracy
    """
    return torch.mean((torch.argmax(y_probs, dim=1) == y_true).to(dtype=torch.float64))


def dirichlet_prior_network_uncertainty(logits, epsilon=1e-10):
    """
    @param logits: model output
    @param epsilon: small value to add to each prob to avoid math error for taking log(0)
    @return: dictionary of different uncertainty measures
    """
    logits = np.asarray(logits, dtype=np.float64)
    alphas = np.exp(logits)
    alpha0 = np.sum(alphas, axis=1, keepdims=True)
    probs = alphas / alpha0

    conf = np.max(probs, axis=1)

    entropy_of_exp = -np.sum(probs * np.log(probs + epsilon), axis=1)
    expected_entropy = -np.sum(probs * (digamma(alphas + 1) - digamma(alpha0 + 1.0)),
                               axis=1)
    mutual_info = entropy_of_exp - expected_entropy

    epkl = np.squeeze((alphas.shape[1] - 1.0) / alpha0)

    dentropy = np.sum(gammaln(alphas) - (alphas - 1.0) * (digamma(alphas) - digamma(alpha0)),
                      axis=1, keepdims=True) \
               - gammaln(alpha0)

    uncertainty = {'confidence': conf,
                   'entropy_of_expected': entropy_of_exp,
                   'expected_entropy': expected_entropy,
                   'mutual_information': mutual_info,
                   'EPKL': epkl,
                   'differential_entropy': np.squeeze(dentropy),
                   }

    return uncertainty
