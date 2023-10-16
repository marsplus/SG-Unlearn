import os
import sys
sys.path.append('../src')

import glob
import argparse
import numpy as np

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18


import mia_evaluate
from utils import evaluate_accuracy


from scipy.stats import ks_2samp
from collections import defaultdict

import pdb
import warnings
warnings.simplefilter(action='ignore', category=Warning)


def evaluate_model(model=None, data=None, batch_size=128, seed=1, device='cuda:0'):
    """
        Evaluate the model on the particular data.
    """
    ret = defaultdict(float)
    RNG = torch.Generator().manual_seed(seed)
    
    retain_dataset = data['retain']
    test_dataset   = data['test']
    val_dataset    = data['val']
    forget_dataset = data['forget']
    
    retain_loader = torch.utils.data.DataLoader(retain_dataset, batch_size=batch_size, shuffle=True, num_workers=2, generator=RNG)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, generator=RNG)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2, generator=RNG)
    forget_loader = torch.utils.data.DataLoader(forget_dataset, batch_size=batch_size, shuffle=True, num_workers=2, generator=RNG)

    ret['test accuracy'] = evaluate_accuracy(model, test_loader, device=device)
    ret['retain accuracy'] = evaluate_accuracy(model, retain_loader, device=device)
    ret['forget accuracy'] = evaluate_accuracy(model, forget_loader, device=device)
    ret['val accuracy'] = evaluate_accuracy(model, val_loader, device=device)

    ## evaluate MIA
    MIA_acc, MIA_auc, MIA_f1, forget_losses, test_losses = mia_evaluate.simple_mia(model, forget_loader, test_loader, random_state=seed, device=device)
    ret['MIA accuracy'] = MIA_acc
    ret['MIA auc'] = MIA_auc
    ret['MIA f1']  = MIA_f1

    ## evaluate KS statistics
    ks_statistic, kp_value = ks_2samp(forget_losses, test_losses)
    ret['KS statistic'] = ks_statistic
    ret['KS p-value']   = kp_value
    return ret


def get_stats(path_to_ckpts, is_SG=False):
    """
        Evaluate the checkpoints of an experiment with a particular evaluation function
    """
    ## the evaluation metric to decide which hyper-params to choose
    evaluate_func = lambda ret: ret['val accuracy'] - ret['MIA accuracy']
    if not is_SG:
        all_ckpts = [torch.load(f)['evaluation_result'] for f in path_to_ckpts]
    else:
        all_ckpts = [torch.load(f) for f in path_to_ckpts]
    n = len(all_ckpts)
    ret = defaultdict(float)
    for d in all_ckpts:
        for key, value in d.items():
            ret[key] += (value / n)
    eval_metric = evaluate_func(ret)
    return (eval_metric, ret)


def FT_hyperparam_search(args):
    best_metric = float('-inf')
    best = None
    best_param = (None, None)
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        for ep in [5, 10, 15]:
            folder_path = os.path.join(args.baseline_path, f'FT/{args.dataset}/lr_{lr}_nepoch_{ep}')
            path_to_ckpts = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                            if f.endswith('.tar') and os.path.isfile(os.path.join(folder_path, f))]
            if not path_to_ckpts: continue
            eval_metric, ret = get_stats(path_to_ckpts)
            if eval_metric > best_metric:
                best_metric = eval_metric
                best = ret
                best_param = (lr, ep)
    print(best_param)
    return best


def GA_hyperparam_search(args):
    best_metric = float('-inf')
    best = None
    best_param = (None, None)
    for lr in [0.000001, 0.00001, 0.0001, 0.001]:
        for ep in [5, 10, 15]:
            if lr == 0.000001:
                folder_path = os.path.join(args.baseline_path, f'GA/{args.dataset}/lr_0.000001_nepoch_{ep}')
            elif lr == 0.00001:
                folder_path = os.path.join(args.baseline_path, f'GA/{args.dataset}/lr_0.00001_nepoch_{ep}')
            elif lr == 0.0001:
                folder_path = os.path.join(args.baseline_path, f'GA/{args.dataset}/lr_0.0001_nepoch_{ep}')
            else:
                folder_path = os.path.join(args.baseline_path, f'GA/{args.dataset}/lr_0.001_nepoch_{ep}')
            path_to_ckpts = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                            if f.endswith('.tar') and os.path.isfile(os.path.join(folder_path, f))]
            if not path_to_ckpts: continue
            eval_metric, ret = get_stats(path_to_ckpts)
            if eval_metric > best_metric:
                best_metric = eval_metric
                best = ret
                best_param = (lr, ep)
    print(best_param)
    return best


def fisher_hyperparam_search(args):
    best_metric = float('-inf')
    best = None
    best_param = None
    for alpha in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
        if alpha == 1e-9:
            folder_path = os.path.join(args.baseline_path, f'{args.baseline}/{args.dataset}/alpha_1e-9')
        elif alpha == 1e-8:
            folder_path = os.path.join(args.baseline_path, f'{args.baseline}/{args.dataset}/alpha_1e-8')
        elif alpha == 1e-7:
            folder_path = os.path.join(args.baseline_path, f'{args.baseline}/{args.dataset}/alpha_1e-7')
        elif alpha == 1e-6:
            folder_path = os.path.join(args.baseline_path, f'{args.baseline}/{args.dataset}/alpha_1e-6')
        elif alpha == 1e-5:
            folder_path = os.path.join(args.baseline_path, f'{args.baseline}/{args.dataset}/alpha_1e-5')
        path_to_ckpts = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                        if f.endswith('.tar') and os.path.isfile(os.path.join(folder_path, f))]
        if not path_to_ckpts: continue
        eval_metric, ret = get_stats(path_to_ckpts)
        if eval_metric > best_metric:
            best_metric = eval_metric
            best = ret
            best_param = alpha
    print(best_param)
    return best


def retrain_hyperparam_search(args):
    best_metric = float('-inf')
    best = None
    best_param = (None, None)
    if args.dataset == 'svhn':
        folder_path = os.path.join(args.baseline_path, f'retrain/{args.dataset}/lr_0.1_nepoch_100')
    else:
        folder_path = os.path.join(args.baseline_path, f'retrain/{args.dataset}/lr_0.1_nepoch_200')
    path_to_ckpts = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                    if f.endswith('.tar') and os.path.isfile(os.path.join(folder_path, f))]
    if not path_to_ckpts: raise ValueError("Empty retrain data.")
    eval_metric, ret = get_stats(path_to_ckpts)
    if eval_metric > best_metric:
        best_metric = eval_metric
        best = ret
    print(best_param)
    return best


def SG_hyperparam_search(args):
    dim = 100 if args.dataset == 'cifar100' else 10
    best_metric = float('-inf')
    best = None
    best_param = None
    for ep in [5, 10, 15, 20, 25, 30]:
        path_to_ckpts = glob.glob(os.path.join(
            args.baseline_path,
            f"{args.dataset}/eval_num_epoch_{ep}_cv_3_dim_{dim}_atts_{args.attacker_strength}_seed_*.pth"
        ))
        if not path_to_ckpts: continue
        eval_metric, ret = get_stats(path_to_ckpts, is_SG=True)
        if eval_metric > best_metric:
            best_metric = eval_metric
            best = ret
            best_param = ep
    print(best_param)
    return best
    # dim = 100 if args.dataset == 'cifar100' else 10
    # path_to_ckpts = glob.glob(os.path.join(args.baseline_path, f'{args.dataset}/eval_num_epoch_{args.num_epoch}_cv_3_dim_{dim}_atts_{args.attacker_strength}_seed_*_{args.dataset}.pth'))
    # n = len(path_to_ckpts)
    # ret = defaultdict(float)
    # for f in path_to_ckpts:
    #     d = torch.load(f)
    #     for key in d.keys():
    #         ret[key] += (d[key] / n)
    # return ret
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--baseline_path', type=str, default=".")
    parser.add_argument('--baseline', type=str, default='FT')
    parser.add_argument('--num_epoch', type=int, default=5)
    parser.add_argument('--attacker_strength', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()

    if args.baseline == 'FT':
        best_ret = FT_hyperparam_search(args)
    elif args.baseline == 'GA':
        best_ret = GA_hyperparam_search(args)
    elif args.baseline in ['wfisher', 'fisher_new']:
        best_ret = fisher_hyperparam_search(args)
    elif args.baseline == 'retrain':
        best_ret = retrain_hyperparam_search(args)
    elif args.baseline == 'SG':
        best_ret = SG_hyperparam_search(args)

    best_ret['time'] /= 60.0
    for key, value in best_ret.items():
        if 'losses' not in key:
            print(f"{key}: {value:.4f}")
    print(f"|forget - test|: {np.abs(best_ret['forget accuracy'] - best_ret['test accuracy']):.4f}")
    print('\n')
    