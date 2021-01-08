import os

import matplotlib.pyplot as plt
import json
import numpy as np

SAVE_DIR = "resources/graphics"
JSON_DIR = "resources/json"
os.makedirs(SAVE_DIR, exist_ok=True)


if __name__ == '__main__':
    titlevars = {"accuracy": "Accuracy",
                 "auroc_de": "AUROC(Differential Entropy)",
                 "auroc_mi": "AUROC(Mutual Information)",
                 "id_precision": "In-Domain-Precision",
                 "ood_precision": "Out-Of-Domain-Precision",
                 "loss": "Loss"}

    id_datasets = ["CIFAR10"]
    ood_datasets = ["SVHN", "TIM", "Random"]
    for var in ["accuracy", "auroc_de", "auroc_mi", "id_precision", "ood_precision", "loss"]:

        fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(12, 6))
        fig.suptitle(f"Test-{titlevars[var]} im Epochenverlauf für unterschiedliche OOD-Datensätze", fontsize=16)
        epoch_ticks = [5, 10, 15, 20]
        plt.xticks(epoch_ticks)

        epoch_list = range(1, 21)

        for id_dataset in id_datasets:
            values = {}
            mean = {}
            std = {}
            for ood_dataset in ood_datasets:
                values[ood_dataset] = np.array([])
                for i in range(1, 11):
                    with open(f"{JSON_DIR}/{id_dataset}-{ood_dataset}-{i}.json") as f:
                        data = json.load(f)
                        if len(values[ood_dataset]) == 0:
                            values[ood_dataset] = np.array(data[f"test_{var}"])
                        else:
                            values[ood_dataset] = np.vstack((values[ood_dataset], np.array(data[f"test_{var}"])))
                mean[ood_dataset] = np.mean(values[ood_dataset], 0)
                std[ood_dataset] = np.std(values[ood_dataset], 0)

            axs[0].set_ylabel(var)
            for i, ax in enumerate(axs):
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(color='#dddddd')
                ax.set_xlabel('epoch')
                ax.set_title(ood_datasets[i])

                ci = 2 * std[ood_datasets[i]]
                ax.plot(epoch_list, mean[ood_datasets[i]], linewidth=2)
                ax.fill_between(epoch_list, (mean[ood_datasets[i]] - ci), (mean[ood_datasets[i]] + ci), color='b',
                                alpha=.1)
            plt.savefig(f"{SAVE_DIR}/{var}-all")
            print(f"saved to {SAVE_DIR}/{var}-all.png")

    plt.clf()
    with open(f"{JSON_DIR}/CIFAR10-SVHN-forward-kld.json") as f:
        data = json.load(f)
        fig, ax = plt.subplots()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(color='#dddddd')
        ax.set_xlabel('epoch')

        ax.plot(epoch_list, data["test_id_precision"], linewidth=2)
    plt.savefig(f"{SAVE_DIR}/CIFAR10-SVHN-FKLD-ID-Precision")
    print(f"saved to {SAVE_DIR}/CIFAR10-SVHN-FKLD-ID-Precision.png")
