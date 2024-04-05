#!/bin/bash


device_id=$1   # specify GPU id here
dataset=$2     # specify dataset

dl=0.01
ne=20
cv=3
clas=SVM
output_dir="../result/adv_mini_batch/${dataset}/"
mkdir -p $output_dir

if [ "$dataset" == "cifar10" ]; then
	d=10
elif [ "$dataset" == "cifar100" ]; then
	d=100
else
	d=10
fi

for seed in {1..10}
do
	python main.py
	--device_id=${device_id} \
	--num_epoch=$ne \
	--cv=$cv \
	--dim=$d \
	--output_dir=${output_dir} \
	--seed=$seed \
	--dataset=$dataset \
	--save_checkpoint=1
	# >> ${output_dir}/log_SG_${dataset}*ne*${ne}*seed*${seed}.txt
done
