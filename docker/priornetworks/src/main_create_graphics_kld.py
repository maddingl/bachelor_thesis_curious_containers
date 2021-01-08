import os

import matplotlib.pyplot as plt
import torch

from utils.loss import dirichlet_kl_divergence

SAVE_DIR = "resources/graphics"
os.makedirs(SAVE_DIR, exist_ok=True)

if __name__ == '__main__':
    target_alphas = torch.tensor(
        [[1., 1., 1.],
         [1., 1., 101.]])

    fklds_ood = {}
    fklds_id = {}
    rklds_ood = {}
    rklds_id = {}

    xs = [1, 2, 3]

    for x in xs:
        fklds_ood[x] = []
        fklds_id[x] = []
        rklds_ood[x] = []
        rklds_id[x] = []

        for i in range(20, 500):
            alphas = torch.tensor(
                [[float(x), float(x), float(i)],
                 [float(x), float(x), float(i)]])

            fkld = dirichlet_kl_divergence(alphas, target_alphas).tolist()
            fklds_ood[x].append(fkld[0])
            fklds_id[x].append(fkld[1])

            rkld = dirichlet_kl_divergence(target_alphas, alphas).tolist()
            rklds_ood[x].append(rkld[0])
            rklds_id[x].append(rkld[1])

    for x in xs:
        plt.clf()
        plt.figure(figsize=(3.5, 3.5))
        plt.plot(range(20, 500), fklds_id[x], label="forward KLD")
        plt.plot(range(20, 500), rklds_id[x], label="reverse KLD")
        plt.xlabel(r'$\alpha_3$', labelpad=2)
        plt.legend()
        plt.savefig(f"{SAVE_DIR}/kld-{x}")
