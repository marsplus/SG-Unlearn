#!/bin/bash
ne=30
cv=3
d=10
clas=SVM
with_attacker=0
dataset="svhn"
output_dir="../result/SG_diff_strength_mini_batch_class/${dataset}"
model_path="/code/Unlearn-Bench/examples/results/SVHN/ResNet18/EmpiricalRiskMinimization/pretrain/name_vanilla_train_seed_2/pretrain_checkpoint.pt"
mkdir -p $output_dir

# atts=$2
# for atts in 2 5 0
for atts in 1
do    
    for seed in {1..10}
    do  
        python main_classwise.py \
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
            --baseline_mode=0 \
            --model_path=$model_path
    done
done
