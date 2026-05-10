import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataloader import get_dataloaders

if __name__ == '__main__':
    for dataset in ['mnist', 'cifar10', 'celeba']:
        train_loader, test_loader = get_dataloaders(dataset=dataset)
        images, labels = next(iter(test_loader))
        print(f"{dataset}: {images.shape}, range: {images.min().item():.1f} to {images.max().item():.1f}")