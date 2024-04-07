#!/bin/bash

usage="Usage: $0 <method> <dataset> <device_id>\n
<method>: The method to evaluate. Options include 'SG', 'retrain', etc.\n
<dataset>: The dataset to evaluate on. Example datasets: 'cifar10', 'cifar100'.\n
<device_id>: The device ID to run the evaluation on. For example, '0' for the first GPU.\n"

## Example usage: ./eval.sh SG cifar10 0; this evaluates SG on cifar10 using GPU 0

# Check if exactly three arguments are given, else print the usage
if [ "$#" -ne 3 ]; then
    echo -e $usage
    exit 1
fi

method=$1
dataset=$2
device_id=$3
checkpoint_path="../result/SG_diff_strength_mini_batch"

echo $method
python evaluate.py --dataset ${dataset} --checkpoint_path ${checkpoint_path} --method ${method} --device_id ${device_id}
