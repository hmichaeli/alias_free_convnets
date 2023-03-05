# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# convnext_LICENSE.txt file in the root directory of this source tree.


from configparser import Interpolation
import os
import torch
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    # add cifar 10
    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNET200':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 200
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # add resnet imagenet transformation - from aps code:
        if args.resnet_default_aug:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        elif args.resnet_aug == "random_crop":
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        elif args.resnet_aug == "random_resize_crop":
            transform = transforms.Compose([
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        elif args.resnet_aug == "color_jitter":
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=args.color_jitter, contrast=args.color_jitter, saturation=args.color_jitter),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])


        else:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std,
            )
            if not resize_im:
                transform.transforms[0] = transforms.RandomCrop(
                    args.input_size, padding=4)

        # add zero padding option
        if args.zero_pad_input > 0:
            transform.transforms.append(transforms.Pad(args.zero_pad_input, fill=0, padding_mode='constant'))

        return transform

    # else - eval
    if args.resnet_default_aug:
        interpolation = transforms.InterpolationMode.BILINEAR
    else:
        interpolation = transforms.InterpolationMode.BICUBIC
        
    
    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=interpolation), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=interpolation),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))

    # add zero padding option
    if args.zero_pad_input > 0:
        t.append(transforms.Pad(args.zero_pad_input, fill=0, padding_mode='constant'))

    if args.eval_crop_shift is not None:
        # create imagent eval standard transformation list of transforms
        t = [
            transforms.Resize(256, interpolation=interpolation),
            transforms.CenterCrop(224 + args.eval_crop_shift * 2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

    return transforms.Compose(t)


def find_dataset_stats(data_path="/home/ehoffer/Datasets/imagenet/"):
    from tqdm import tqdm
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(data_path, transform=transform)
    dataset = torch.utils.data.DataLoader(dataset, batch_size=1024, num_workers=8, shuffle=True)

    means = []
    stds = []
    stds_b = []
    N = 50
    for i,img in enumerate(tqdm(dataset)):
        img = img[0].to("cuda:7")
        means.append(torch.mean(img, dim=(0,2,3)))
        stds.append(torch.std(img, dim=(0,2,3)))
        stds_b.append(torch.std(img, dim=(0, 2, 3), unbiased=True))
        print()
        if i > N: break

    mean = torch.mean(torch.stack(means), dim=0)
    std = torch.mean(torch.stack(stds), dim=0)
    std_b = torch.mean(torch.stack(stds_b), dim=0)

    print(f"mean: {mean}, std: {std}")
    print(f"std biased: {std_b}")
    print()


if __name__ == '__main__':
    find_dataset_stats()
