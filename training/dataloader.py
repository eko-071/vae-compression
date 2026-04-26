import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(data_dir="./data", batch_size=128, num_workers=0):
    transform = transforms.Compose(
        [transforms.ToTensor(),]
    )

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