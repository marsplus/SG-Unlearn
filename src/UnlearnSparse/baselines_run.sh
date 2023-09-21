#!/bin/bash

alpha=1e-8
output_dir="../../result/baselines"
mkdir -p $output_dir

counter=0
for seed in {1..10}
do
    # python -u main_forget.py --save_dir results --unlearn FT  --unlearn_lr 0.01 --unlearn_epochs 10 --gpu 1  --seed ${seed} &
    # python -u main_forget.py --save_dir results --unlearn GA  --unlearn_lr 0.0001 --unlearn_epochs 5 --gpu 2 --seed ${seed} &
    # python -u main_forget.py --save_dir results --unlearn fisher_new --alpha ${alpha} --gpu 0 --seed ${seed} &
    # python -u main_forget.py --save_dir results --unlearn wfisher --num_indexes_to_replace 4500 --alpha ${alpha} --gpu 2 --seed ${seed} &
    python -u main_forget.py --save_dir=${output_dir} --unlearn retrain --unlearn_epochs 160 --unlearn_lr 0.1 --gpu 2 --seed ${seed} &

    counter=$((counter+1))

    # If counter reaches 2, wait for background processes to finish, then reset the counter
    if [ $counter -eq 2 ]; then
        wait
        counter=0
    fi
done


