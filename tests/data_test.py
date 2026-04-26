import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataloader import get_mnist_loaders

if __name__ == '__main__':
    train_loader, test_loader = get_mnist_loaders()

    images, labels = next(iter(train_loader))
    print("Train batch shape:", images.shape)
    print("Pixel range:", images.min().item(), "to", images.max().item())

    images, labels = next(iter(test_loader))
    print("Test batch shape:", images.shape)