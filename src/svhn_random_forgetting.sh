


#!/bin/bash

ne=30
cv=3
d=10
clas=SVM
with_attacker=0
dataset="svhn"
output_dir="../result/SG_diff_strength_mini_batch/${dataset}"
mkdir -p $output_dir


for atts in 1
do    
    for seed in {1..10}
    do  
        python main.py \
            --device_id=$1 \
            --num_epoch=$ne \
            --cv=$cv \
            --dim=$d \
            --output_dir=$output_dir \
            --seed=$seed \
            --dataset=$dataset \
            --save_checkpoint=1 \
            --attacker_strength=$atts \
            --mem_save=10 \
            --baseline_mode=0
    done
done
