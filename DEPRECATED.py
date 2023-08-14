import torch
import torchvision
from torchvision import datasets, transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import matplotlib.pyplot as plt


class cifar_ds10(torchvision.datasets.CIFAR10):
    def __init__(self, root="./data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )
    return trainloader, trainset


def set_albumen_params(mean, std):
    horizontalflip_prob = 0.25
    rotate_limit = 15
    shiftscalerotate_prob = 0.25
    num_holes = 1
    cutout_prob = 0.5
    max_height = 16  # 32/2
    max_width = 16  # 32/2

    transform_train = A.Compose(
        [
            A.PadIfNeeded(min_height=36, min_width=36),
            A.RandomCrop(width=32, height=32, always_apply=False),
            A.HorizontalFlip(p=horizontalflip_prob),
            A.CoarseDropout(
                max_holes=num_holes,
                min_holes=1,
                max_height=max_height,
                max_width=max_width,
                p=cutout_prob,
                fill_value=tuple([x * 255.0 for x in mean]),
                min_height=max_height,
                min_width=max_width,
                mask_fill_value=None,
            ),
            A.Normalize(mean=mean, std=std, p=1.0, always_apply=True),
            ToTensorV2(),
        ]
    )

    transform_valid = A.Compose(
        [
            A.Normalize(
                mean=mean,
                std=std,
                p=1.0,
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ]
    )
    return transform_train, transform_valid


def show_sample(dataset):
    classes = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    dataiter = iter(dataset)

    index = 0
    fig = plt.figure(figsize=(20, 10))
    for i in range(10):
        images, labels = next(dataiter)
        actual = classes[labels]
        image = images.squeeze().to("cpu").numpy()
        ax = fig.add_subplot(2, 5, index + 1)
        index = index + 1
        ax.axis("off")
        ax.set_title(f"\n Label : {actual}", fontsize=10)
        ax.imshow(np.transpose(image, (1, 2, 0)))
        images, labels = next(dataiter)


def tl_ts_mod(transform_train, transform_valid, batch_size=128):
    trainset = cifar_ds10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testset = cifar_ds10(
        root="./data", train=False, download=True, transform=transform_valid
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return trainset, trainloader, testset, testloader


def process_dataset(batch_size=128):
    trl, trs = load_data()

    show_sample(trs)

    mean = list(np.round(trs.data.mean(axis=(0, 1, 2)) / 255, 4))
    std = list(np.round(trs.data.std(axis=(0, 1, 2)) / 255, 4))

    transform_train, transform_valid = set_albumen_params(mean, std)
    trainset_mod, trainloader_mod, testset_mod, testloader_mod = tl_ts_mod(
        transform_train, transform_valid, batch_size=batch_size
    )

    return trainset_mod, trainloader_mod, testset_mod, testloader_mod