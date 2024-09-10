import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms

def load_dataset(name: str):
    if name == 'cifar10':
        return load_cifar10()
    elif name == 'fashion_mnist':
        return load_fashion_mnist()
    elif name == 'svhn':
        return load_svhn()
    elif name == 'mnist':
        return load_mnist()
    else:
        raise NotImplementedError(f"Dataset {name} not implemented")


def load_fashion_mnist():
    train_set = datasets.FashionMNIST(root="./data", train=True, download=True,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]))
    test_set = datasets.FashionMNIST(root="./data", train=False, download=True,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]))
    return train_set, test_set


def load_mnist():
    train_set = datasets.MNIST(root="./data", download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    test_set = datasets.MNIST(root="./data", download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    return train_set, test_set

def load_svhn():
    train_set = datasets.SVHN(root="./data", download=True, split='train', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    test_set = datasets.SVHN(root="./data", download=True, split='test', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    return train_set, test_set


def load_cifar10():
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    return train_set, test_set


class PartitionedDataset(Dataset):
    def __init__(self, dataset, indexes):
        self.dataset = dataset
        self.indexes = list(indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        image, label = self.dataset[self.indexes[item]]
        return image, label


class RandomSampledDataset2(Dataset):
    def __init__(self, dataset, indexes, q=1.0):
        self.dataset = dataset
        self.indexes = list(indexes)
        self.length = int(len(self.indexes) * q)
        self.count = 0
        self.random_indexes = np.random.choice(self.indexes, self.length, replace=False)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image, label = self.dataset[self.random_indexes[item]]
        self.count += 1
        if self.count == self.length:
            self.reset()
        return image, label

    def reset(self):
        self.count = 0
        self.random_indexes = np.random.choice(self.indexes, self.length, replace=False)


