from utils.graphics import Graphics
from utils.normalization_parameter_calculator import calc_mean_std
from utils.standardised_datasets import TinyImageNet
from torchvision import datasets, transforms

if __name__ == '__main__':
    # graphics = Graphics("test")
    # graphics.plot_loss()
    # graphics.plot_accuracy()

    data_dir = "resources/data"
    dataset = TinyImageNet(root=f"{data_dir}/TIM",
                                     transform=transforms.Compose([transforms.ToTensor()]),
                                     target_transform=None,
                                     split="train")

    print(calc_mean_std(dataset, 3, 1000))
