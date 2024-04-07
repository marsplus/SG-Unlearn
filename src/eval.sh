

# method=$1
# dataset=$2
dataset=$1
device_id=$2
checkpoint_path="../result/SG_diff_strength_mini_batch"

for method in SG
do
    echo $method
    python evaluate.py --dataset ${dataset} --checkpoint_path ${checkpoint_path} --method ${method} --device_id ${device_id}
done
