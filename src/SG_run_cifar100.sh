#!/bin/bash

nepoch=15
cv=3
d=101
clas=SVM
output_dir="../result/SG_data_cifar100-${nepoch}epoch"
mkdir -p $output_dir

counter=0
dev_id=0
for seed in {1..10}
do    
    python cifar.py --device_id=$dev_id --num_epoch=$nepoch --cv=$cv --dim=$d --output_dir=${output_dir} --seed=$seed  --save_checkpoint=1  --dataset=cifar100 --output_sc_fname=SG_data_cifar100_cv_${cv}_dim_${dim}_epoch_${nepoch}_seed_${seed} 
    
#     counter=$((counter+1))
    
#     # If counter reaches 2, wait for background processes to finish, then reset the counter
#     if [ $counter -eq 2 ]; then
#         wait
#         counter=0
#     fi
done
