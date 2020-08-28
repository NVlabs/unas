# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for UNAS. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


"""Code for getting the data loaders."""

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
from util import utils
from torch._utils import _accumulate


def get_loaders(args, mode='eval', dataset=None):
    """Get data loaders for required dataset."""
    if dataset is None:
        dataset = args.dataset
    if dataset == 'imagenet':
        return get_imagenet_loader(args, mode)
    else:
        if mode == 'search':
            return get_loaders_search(args)
        elif mode == 'eval':
            return get_loaders_eval(dataset, args)

################################################################################
# CIFAR-10 / tinyImageNet
################################################################################


def get_loaders_eval(dataset, args):
    """Get train and valid loaders for cifar10/tiny imagenet."""

    if dataset == 'cifar10':
        num_classes = 10
        train_transform, valid_transform = _data_transforms_cifar10(args)
        train_data = dset.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(
            root=args.data, train=False, download=True, transform=valid_transform)
    elif dataset == 'cifar100':
        num_classes = 100
        train_transform, valid_transform = _data_transforms_cifar10(args)
        train_data = dset.CIFAR100(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(
            root=args.data, train=False, download=True, transform=valid_transform)

    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False,
        sampler=valid_sampler, pin_memory=True, num_workers=2)

    return train_queue, valid_queue, num_classes


def get_loaders_search(args):
    """Get train and valid loaders for cifar10/tiny imagenet."""

    if args.dataset == 'cifar10':
        num_classes = 10
        train_transform, _ = _data_transforms_cifar10(args)
        train_data = dset.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_transform, _ = _data_transforms_cifar10(args)
        train_data = dset.CIFAR100(
            root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    print('Found %d samples' % (num_train))
    sub_num_train = int(np.floor(args.train_portion * num_train))
    sub_num_valid = num_train - sub_num_train

    sub_train_data, sub_valid_data = my_random_split(
        train_data, [sub_num_train, sub_num_valid], seed=0)
    print('Train: Split into %d samples' % (len(sub_train_data)))
    print('Valid: Split into %d samples' % (len(sub_valid_data)))

    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            sub_train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            sub_valid_data)

    train_queue = torch.utils.data.DataLoader(
        sub_train_data, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler, pin_memory=False, num_workers=4, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        sub_valid_data, batch_size=args.batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler, pin_memory=False, num_workers=4, drop_last=True)

    return train_queue, valid_queue, num_classes


def _data_transforms_cifar10(args):
    """Get data transforms for cifar10."""
    mean = [0.49139968, 0.48215827, 0.44653124]
    std = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if hasattr(args, 'cutout') and args.cutout:
        train_transform.transforms.append(utils.Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, valid_transform


def _data_transforms_tinyimagenet(args):
    """Get data transforms for tinyimagenet."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if args.cutout:
        train_transform.transforms.append(utils.Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, valid_transform


def get_imagenet_loader(args, mode='eval'):
    """Get train/val for imagenet."""
    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    downscale = 1
    val_transform = transforms.Compose([
        transforms.Resize(256//downscale),
        transforms.CenterCrop(224//downscale),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224//downscale),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if mode == 'eval':
        if 'lmdb' in args.data:
            train_data = imagenet_lmdb_dataset(
                traindir, transform=train_transform)
            valid_data = imagenet_lmdb_dataset(
                validdir, transform=val_transform)
        else:
            train_data = dset.ImageFolder(traindir, transform=train_transform)
            valid_data = dset.ImageFolder(validdir, transform=val_transform)

        train_sampler, valid_sampler = None, None
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data)

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            pin_memory=True, num_workers=4, sampler=train_sampler)

        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=False,
            pin_memory=True, num_workers=4, sampler=valid_sampler)
    elif mode == 'search':
        if 'lmdb' in args.data:
            train_data = imagenet_lmdb_dataset(
                traindir, transform=val_transform)
        else:
            train_data = dset.ImageFolder(traindir, val_transform)

        num_train = len(train_data)
        print('Found %d samples' % (num_train))
        sub_num_train = int(np.floor(args.train_portion * num_train))
        sub_num_valid = num_train - sub_num_train

        sub_train_data, sub_valid_data = my_random_split(
            train_data, [sub_num_train, sub_num_valid], seed=0)
        print('Train: Split into %d samples' % (len(sub_train_data)))
        print('Valid: Split into %d samples' % (len(sub_valid_data)))

        train_sampler, valid_sampler = None, None
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                sub_train_data)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                sub_valid_data)

        train_queue = torch.utils.data.DataLoader(
            sub_train_data, batch_size=args.batch_size,
            sampler=train_sampler, shuffle=(train_sampler is None),
            pin_memory=False, num_workers=4, drop_last=True)

        valid_queue = torch.utils.data.DataLoader(
            sub_valid_data, batch_size=args.batch_size,
            sampler=valid_sampler, shuffle=(valid_sampler is None),
            pin_memory=False, num_workers=4, drop_last=True)

    return train_queue, valid_queue, 1000

################################################################################


def my_random_split(dataset, lengths, seed=0):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!")
    g = torch.Generator()
    g.manual_seed(seed)

    indices = torch.randperm(sum(lengths), generator=g)
    return [torch.utils.data.Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


################################################################################
# ImageNet - LMDB
################################################################################

import io
import os
try:
    import lmdb
except:
    pass
import torch
from torchvision import datasets
from PIL import Image


def lmdb_loader(path, lmdb_data):
    # In-memory binary streams
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path.encode('ascii'))
    img = Image.open(io.BytesIO(bytedata))
    return img.convert('RGB')


def imagenet_lmdb_dataset(
        root, transform=None, target_transform=None,
        loader=lmdb_loader):
    if root.endswith('/'):
        root = root[:-1]
    pt_path = os.path.join(
        root + '_faster_imagefolder.lmdb.pt')
    lmdb_path = os.path.join(
        root + '_faster_imagefolder.lmdb')
    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        print('Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    else:
        data_set = datasets.ImageFolder(
            root, None, None, None)
        torch.save(data_set, pt_path, pickle_protocol=4)
        print('Saving pt to {}'.format(pt_path))
        print('Building lmdb to {}'.format(lmdb_path))
        env = lmdb.open(lmdb_path, map_size=1e12)
        with env.begin(write=True) as txn:
            for path, class_index in data_set.imgs:
                with open(path, 'rb') as f:
                    data = f.read()
                txn.put(path.encode('ascii'), data)
    data_set.lmdb_data = lmdb.open(
        lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False,
        meminit=False)
    # reset transform and target_transform
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    data_set.loader = lambda path: loader(path, data_set.lmdb_data)
    return data_set
