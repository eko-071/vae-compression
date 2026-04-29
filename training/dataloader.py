import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DATASET_INFO = {
    'mnist': {'channels': 1, 'height': 28, 'width': 28},
    'cifar10': {'channels': 3, 'height': 32, 'width': 32},
}

def get_input_dim(dataset):
    info = DATASET_INFO[dataset]
    return info['channels'] * info['height'] * info['width']

def get_dataloaders(dataset='mnist', data_dir="./data", batch_size=128, num_workers=0):
    if dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(),])

        train_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )

        test_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(),])

        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader