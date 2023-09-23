import os
import sys
sys.path.append('../src')

import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid

from utils import random_split
from models import DefenderOPT

import torch.nn as nn
# from resnet import resnet18
from torchvision.models import resnet18

import pdb

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(args):
    num_workers = 4
    DEVICE = f'cuda:{args.device_id}' if torch.cuda.is_available() else "cpu"
    ## download and pre-process CIFAR10
    if args.dataset == 'cifar10':
        transform = transforms.Compose(
                    [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            )
    elif args.dataset == 'cifar100':
        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])


    if args.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
        held_out = torchvision.datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)
        args.num_class = 10
    elif args.dataset == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(root="../data", train=True, download=True, transform=transform_train)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
        held_out = torchvision.datasets.CIFAR100(root="../data", train=False, download=True, transform=transform_test)
        args.num_class = 100

    test_set, val_set = random_split(held_out, [0.5, 0.5], generator=RNG)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_set,   batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    ## construct retain and forget sets
    forget_set, retain_set = random_split(train_set, [0.1, 0.9], generator=RNG)
    forget_loader = torch.utils.data.DataLoader(
        forget_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, generator=RNG)
    retain_loader = torch.utils.data.DataLoader(
        retain_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, generator=RNG)
    

    ## the unlearned model
    if args.arch == 'resnet18':
        if args.dataset == 'cifar10':
            local_path = args.model_path if args.model_path else os.path.join(ROOT_DIR, "../models/cifar10_resnet18_ckpt.pth")
        elif args.dataset == 'cifar100':
            local_path = args.model_path if args.model_path else os.path.join(ROOT_DIR, "../models/cifar100_resnet18_ckpt_native.pth")
        else:
            raise ValueError("Unknow dataset.\n")
    else:
        raise ValueError("Unknow network architecture.\n")
        
    weights_pretrained = torch.load(local_path, map_location=DEVICE)
    model_ft = resnet18(num_classes=args.num_class)
    ## change the first conv layer for smaller images
    model_ft.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
    model_ft.load_state_dict(weights_pretrained)
    model_ft.to(DEVICE)

    ## define the defender and run the unlearning algo.
    defender = DefenderOPT(retain_loader, 
                           forget_loader, 
                           val_loader, 
                           test_loader, 
                           num_class=args.num_class,
                           dim=args.dim,
                           cv=args.cv,
                           num_epoch=args.num_epoch,
                           batch_size=args.batch_size,
                           baseline_mode=args.baseline_mode,
                           regular_type=args.regular_type,
                           regular_strength=args.regular_strength,
                           defender_lr=args.defender_lr,
                           attacker_lr=args.attacker_lr,
                           with_attacker=args.with_attacker,
                           attacker_reg=args.attacker_reg,
                           wasserstein_coeff=args.w_coeff,
                           output_sc_fname=args.output_sc_fname,
                           seed=args.seed,
                           weight_decay=args.weight_decay,
                           momentum=args.momentum,
                           device_id=args.device_id,
                           fine_tune=args.fine_tune,
                           att_classifier=args.att_classifier)
    defender.unlearn(model_ft, args.output_dir)
    
    if args.save_checkpoint:
        checkpoint = model_ft.state_dict()
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f'SGcheckpoint_num_epoch_{args.num_epoch}_cv_{args.cv}_dim_{args.dim}_seed_{args.seed}_{args.dataset}.pth')
        torch.save(checkpoint, output_path)

        ## save the data
        output_data = {
            'retain': retain_loader.dataset,
            'test': test_loader.dataset,
            'val': val_loader.dataset,
            'forget': forget_loader.dataset}
        torch.save(output_data, os.path.join(args.output_dir, f'SGdata_seed_{args.seed}_{args.dataset}.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--defender_lr', type=float, default=0.01)
    parser.add_argument('--attacker_lr', type=float, default=0.01)
    parser.add_argument('--dim', type=int, default=1)
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--num_epoch', type=int, default=30)
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--with_attacker', type=int, default=1)
    parser.add_argument('--baseline_mode', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--attacker_reg', type=float, default=0)
    parser.add_argument('--regular_type', type=str, default='l2')
    parser.add_argument('--regular_strength', type=float, default=0.5)
    parser.add_argument('--w_coeff', type=float, default=0.5)
    parser.add_argument('--output_sc_fname', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr_scheduler', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--ts_baseline', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--fine_tune', type=int, default=1)
    parser.add_argument('--log_file', type=str, default='output.log')
    parser.add_argument('--att_classifier', type=str, default='SVM')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--save_checkpoint', type=int, default=0)
    parser.add_argument('--model_path', type=str, default=None)
    # parser.add_argument('--ckpt_path', type=str)
    args = parser.parse_args()

    RNG = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
