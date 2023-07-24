import os
import sys
import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

import utils
import vision_transformer as vits
from src.data.custom_dataset import EncodingsToLabels


def generate_embeddings(args):
    utils.init_distributed_mode(args)
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    model.cuda()
    model.eval()
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    # ============ preparing data ... ============
    train_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # pth_transforms.Normalize((0.4915, 0.4823, 0.4468),
        #                          (0.2470, 0.2435, 0.2616)),
    ])

    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # pth_transforms.Normalize((0.4915, 0.4823, 0.4468),
        #                          (0.2470, 0.2435, 0.2616)),
    ])

    dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=train_transform)
    dataset_val = datasets.CIFAR10(args.data_path, train=False, download=True, transform=val_transform)

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

    n = args.n_last_blocks
    encodings = []
    for images, _ in tqdm(val_loader):
        images = images.to("cuda")
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(images, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if args.avgpool_patchtokens:
                    output = torch.cat(
                        (output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(images)

        encodings.append(output.detach().to("cpu"))
    encodings = torch.cat(encodings)
    print(encodings.shape, type(encodings[0]))

    n = args.n_last_blocks
    encodings_train = []
    for images, _ in tqdm(train_loader):
        images = images.to("cuda")
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(images, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if args.avgpool_patchtokens:
                    output = torch.cat(
                        (output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(images)

        encodings_train.append(output.detach().to("cpu"))
    encodings_train = torch.cat(encodings_train)
    print(encodings_train.shape, type(encodings_train[0]))

    embeddings_dataset_train = EncodingsToLabels(encodings_train, dataset_train.targets)
    embeddings_dataset_test = EncodingsToLabels(encodings, dataset_val.targets)

    torch.save(embeddings_dataset_test, 'cifar10_emb_test_dataset.pt')
    torch.save(embeddings_dataset_train, 'cifar10_emb_train_dataset.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
                        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights',
                        default='./output/dino_deitsmall16_pretrain_full_checkpoint.pth',
                        type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='./datasets/cifar10/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="./output/",
                        help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=10, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

    args = parser.parse_args()
    generate_embeddings(args)
