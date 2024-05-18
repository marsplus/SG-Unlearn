#!/bin/bash

ne=30
cv=3
d=100
clas=SVM
with_attacker=1
dataset="cifar100"
output_dir="../result/SG_diff_strength_mini_batch_class/${dataset}"
mkdir -p $output_dir

# atts=$2
# for atts in 0.05 0.1 0.25 0.5 2 5
for atts in 1
do  
    for seed in {1..10}
    do  
        python main_classwise.py \
            --device_id="$1" \
            --num_epoch=$ne \
            --num_class=100 \
            --cv=$cv \
            --dim=$d \
            --output_dir=$output_dir \
            --seed="$seed" \
            --dataset=$dataset \
            --save_checkpoint=1 \
            --attacker_strength=$atts \
            --mem_save=10 \
            --baseline_mode=0 \
            --augmentation
    done
done