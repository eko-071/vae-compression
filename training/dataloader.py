import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

DATASET_INFO = {
    'mnist': {'channels': 1, 'height': 28, 'width': 28},
    'cifar10': {'channels': 3, 'height': 32, 'width': 32},
    'celeba': {'channels': 3, 'height': 64, 'width': 64}
}

class CelebA(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = os.path.join(root, 'celeba')
        self.img_dir = os.path.join(self.root, 'img_align_celeba')
        self.transform = transform

        split_map = {'train': 0, 'val': 1, 'test': 2}
        split_id = split_map[split]


        partition_file = os.path.join(self.root, 'list_eval_partition.txt')
        self.filenames = []
        with open(partition_file, 'r') as f:
            for line in f:
                fname, sid = line.strip().split()
                if int(sid) == split_id:
                    self.filenames.append(fname)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0 

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
    elif dataset == 'celeba':
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ])

        train_dataset = CelebA(
            root=data_dir,
            split='train',
            transform=transform,
        )
        
        test_dataset = CelebA(
            root=data_dir,
            split='test',
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