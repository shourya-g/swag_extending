import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FilteredCIFAR10(Dataset):
    """
    Fast CIFAR-10 class filter.

    Uses dataset.targets directly instead of calling dataset[idx],
    so it does not transform every image just to inspect labels.
    """

    def __init__(self, dataset, allowed_classes, remap_labels=True):
        self.dataset = dataset
        self.allowed_classes = list(allowed_classes)
        self.allowed_set = set(self.allowed_classes)
        self.remap_labels = remap_labels

        self.label_map = {
            original_label: new_label
            for new_label, original_label in enumerate(self.allowed_classes)
        }

        self.indices = [
            idx for idx, label in enumerate(dataset.targets)
            if label in self.allowed_set
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.dataset[real_idx]

        if self.remap_labels:
            label = self.label_map[label]

        return image, label


def get_cifar10_transforms(
    image_size: int = 32,
    normalization: str = "cifar10",
    augment: bool = True,
):
    """
    Creates train/test transforms.

    For ResNet CIFAR-10:
        image_size=32
        normalization=cifar10

    For pretrained ViT:
        image_size=224
        normalization=imagenet
    """

    if normalization == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif normalization == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    if image_size == 32:
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    else:
        if augment:
            train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return train_transform, test_transform


def get_cifar10_loaders(
    batch_size: int,
    num_workers: int,
    train_classes=None,
    test_classes=None,
    image_size: int = 32,
    normalization: str = "cifar10",
    augment: bool = True,
):
    """
    Standard CIFAR-10 loaders with optional class filtering.

    Normal ResNet run:
        train_classes=None
        test_classes=None
        image_size=32
        normalization=cifar10

    ViT run:
        image_size=224
        normalization=imagenet

    OOD 5-class run:
        train_classes=[0,1,2,3,4]
        test_classes=[0,1,2,3,4]
    """

    train_transform, test_transform = get_cifar10_transforms(
        image_size=image_size,
        normalization=normalization,
        augment=augment,
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="data/raw",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="data/raw",
        train=False,
        download=True,
        transform=test_transform,
    )

    if train_classes is not None:
        train_dataset = FilteredCIFAR10(
            train_dataset,
            allowed_classes=train_classes,
            remap_labels=True,
        )

    if test_classes is not None:
        test_dataset = FilteredCIFAR10(
            test_dataset,
            allowed_classes=test_classes,
            remap_labels=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


def get_cifar10_full_test_loader(
    batch_size: int,
    num_workers: int,
    image_size: int = 32,
    normalization: str = "cifar10",
):
    """
    Full CIFAR-10 test loader with original labels 0-9.

    Used for OOD evaluation.
    """

    _, test_transform = get_cifar10_transforms(
        image_size=image_size,
        normalization=normalization,
        augment=False,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="data/raw",
        train=False,
        download=True,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return test_loader