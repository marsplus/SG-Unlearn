# SG-Unlearn
* The code to replicate the experimental results presented in the paper [Adversarial Machine Unlearning](https://arxiv.org/abs/2406.07687))
* Install needed packages: `conda env create -f environment.yml`
* Install an old version of pytorch: `conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia`
* CIFAR-10 random forgetting: `cd src/; chmod +x cifar10_random_forgetting.sh; ./cifar10_random_forgetting.sh 0`, where `0` is the GPU id. 
* CIFAR-100 random forgetting: `cd src/; chmod +x cifar100_random_forgetting.sh; ./cifar100_random_forgetting.sh 0`.
