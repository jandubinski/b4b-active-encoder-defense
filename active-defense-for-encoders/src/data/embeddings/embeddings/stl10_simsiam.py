import os
import sys
import argparse
import json
from pathlib import Path

import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as pth_transforms

from tqdm import tqdm
import torchvision.models as models
from src.data.custom_dataset import EncodingsToLabels


def generate_embeddings(args):

    # ============ building network ... ============
    print("=> loading model '{}'".format(args.arch))

    victim_model = models.__dict__[args.arch]()
    checkpoint = torch.load(
        "checkpoint_0099.pth.tar",
        map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith('module.encoder') and not k.startswith(
                'module.encoder.fc'):
            # remove prefix
            state_dict[k[len("module.encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    print("state dict", state_dict.keys())
    victim_model.load_state_dict(state_dict, strict=False)
    victim_model.fc = torch.nn.Identity()

    victim_model.cuda()
    victim_model.eval()
    # load weights to evaluate
    # utils.load_pretrained_weights(victim_model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    # ============ preparing data ... ============
    # train_transform = pth_transforms.Compose([
    #     pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    #     pth_transforms.RandomResizedCrop(224),
    #     pth_transforms.RandomHorizontalFlip(),
    #     pth_transforms.ToTensor(),
    #     # pth_transforms.Normalize((0.4914, 0.4822, 0.4465),
    #     #                          (0.2023, 0.1994, 0.2010))
    #     # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #     # pth_transforms.Normalize((0.4915, 0.4823, 0.4468),
    #     #                          (0.2470, 0.2435, 0.2616)),
    # ])

    # This transform train for train and test data is a bit weird, probably mistake in code of stealing, but I proceed to be consistent
    transform_train = pth_transforms.Compose([
        pth_transforms.Resize(224),
        # transforms.RandomCrop(32, padding=4),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # val_transform = pth_transforms.Compose([
    #     pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    #     pth_transforms.CenterCrop(224),
    #     pth_transforms.ToTensor(),
    #     # pth_transforms.Normalize((0.4914, 0.4822, 0.4465),
    #     #                      (0.2023, 0.1994, 0.2010))
    #     # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #     # pth_transforms.Normalize((0.4915, 0.4823, 0.4468),
    #     #                          (0.2470, 0.2435, 0.2616)),
    # ])

    dataset_train = datasets.STL10(args.data_path, split="train", download=True, transform=transform_train)
    dataset_val = datasets.STL10(args.data_path, split="test", download=True, transform=transform_train)

    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    # sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        # sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    encodings = []
    for images, _ in tqdm(val_loader):
        images = images.to("cuda")
        with torch.no_grad():
            output = victim_model(images)

        encodings.append(output.detach().to("cpu"))
    encodings = torch.cat(encodings)
    print(encodings.shape, type(encodings[0]))

    encodings_train = []
    for images, _ in tqdm(train_loader):
        images = images.to("cuda")
        with torch.no_grad():
            output = victim_model(images)

        encodings_train.append(output.detach().to("cpu"))
    encodings_train = torch.cat(encodings_train)
    print(encodings_train.shape, type(encodings_train[0]))

    embeddings_dataset_train = EncodingsToLabels(encodings_train, dataset_train.labels)
    embeddings_dataset_test = EncodingsToLabels(encodings, dataset_val.labels)

    torch.save(embeddings_dataset_test, 'stl10_emb_simsiam_test_dataset.pt')
    torch.save(embeddings_dataset_train, 'stl10_emb_simsiam_train_dataset.pt')


if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DIR',
                        help='path to imagenet dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', metavar='N', type=int,
                        # default=4096,
                        default=128,
                        help='mini-batch size (default: 4096), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial (base) learning rate',
                        dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                        metavar='W', help='weight decay (default: 0.)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume', action='store_true',
                        help='resume from checkpoint (if present)')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='dataset for downstream task')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # additional configs:
    parser.add_argument('--pretrained', default='', type=str,
                        help='path to simsiam pretrained checkpoint')
    parser.add_argument('--lars', action='store_true',
                        help='Use LARS')
    parser.add_argument('--epochstrain', default=200, type=int, metavar='N',
                        help='number of epochs victim was trained with')
    parser.add_argument('--epochssteal', default=100, type=int, metavar='N',
                        help='number of epochs stolen model was trained with')
    parser.add_argument('--num_queries', default=50000, type=int, metavar='N',
                        help='number of queries to steal with with')
    parser.add_argument('--losstype', default='mse', type=str,
                        help='Loss function to use.')
    parser.add_argument('--useval', default='False', type=str,
                        help='Use validation set for stealing (only with imagenet)')
    parser.add_argument('--useaug', default='False', type=str,
                        help='Use augmentations with stealing')
    parser.add_argument('--datasetsteal', default='cifar10', type=str,
                        help='dataset used for querying')
    parser.add_argument('--temperature', default=0.1, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--temperaturesn', default=1000, type=float,
                        help='temperature for soft nearest neighbors loss')
    parser.add_argument('--data_path', default='./datasets/stl10/', type=str)
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')

    args = parser.parse_args()
    generate_embeddings(args)
