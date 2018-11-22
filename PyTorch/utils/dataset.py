"""
An utils code for loading dataset
"""
import os

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from datetime import datetime
import utils.imagenet_utils as imagenet_utils
import utils.CIFAR10_utils as CIFAR10_utils


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def get_dataloader(dataset_name, split, batch_size, add_split = None, ratio=-1):

    print ('[%s] Loading %s from %s' %(datetime.now(), split, dataset_name))

    if dataset_name == 'MNIST':

        data_root_list = ['/home/shangyu/MNIST', '/data/datasets/MNIST']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break

        if split == 'train':
            trainset = datasets.MNIST(data_root, train=True, download=False,
                               transform=transforms.Compose([
                                   # transforms.Resize(32),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
            loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        elif split == 'test' or split == 'val':
            testset = datasets.MNIST(data_root, train=False, download=False,
                                transform=transforms.Compose([
                                    # transforms.Resize(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ]))
            loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    elif dataset_name == 'SVHN':
        normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
        if split == 'train':
            trainset = datasets.SVHN(root='/home/shangyu/SVHN', split='train', download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         normalize]))
            loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        elif split == 'test':
            trainset = datasets.SVHN(root='/home/shangyu/SVHN', split='test', download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         normalize]))
            loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        return loader

    elif dataset_name == 'CIFAR10':

        data_root_list = ['/home/shangyu/CIFAR10', '/home/sinno/csy/CIFAR10']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break

        if split == 'train':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            # trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False,
                                                    # transform=transform_train)
            trainset = CIFAR10_utils.CIFAR10(root=data_root, train=True, download=True,
                                                    transform=transform_train, ratio=ratio)
            print ('Number of training instances used: %d' %(len(trainset)))
            loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        elif split == 'test' or split == 'val':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True,
                                                   transform=transform_test)
            loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    elif dataset_name == 'ImageNet':
        traindir = ('./train_imagenet_list.pkl', './classes.pkl', './classes-to-idx.pkl','/data/imagenet/train')
        valdir = ('./val_imagenet_list.pkl', './classes.pkl', './classes-to-idx.pkl', '/data/imagenet/val-pytorch')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if split == 'train':
            trainDataset = imagenet_utils.ImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), ratio=ratio)
            print ('Number of training data used: %d' %(len(trainDataset)))
            loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, \
                                                 num_workers=4 * torch.cuda.device_count(), pin_memory=True)

        elif split == 'val' or split == 'test':
            valDataset = imagenet_utils.ImageFolder(valdir, transforms.Compose([
		        transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
            loader = torch.utils.data.DataLoader(valDataset, batch_size=batch_size, shuffle=True, \
                                                 num_workers=4 * torch.cuda.device_count(), pin_memory=True)

    print ('[%s] Loading finish.' %(datetime.now()))

    return loader


if __name__ == '__main__':

    loader = get_dataloader('MNIST', 'train', 128)
    for batch_idx, (inputs, targets) in enumerate(loader):
        break