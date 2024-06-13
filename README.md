# SG-Unlearn
* The code to replicate the experimental results presented in the paper [Adversarial Machine Unlearning: A Stackelberg Game Approach](https://openreview.net/forum?id=iQIQT88prm&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FConference%2FAuthors%23your-submissions))
* Install needed packages: `conda env create -f environment.yml`
* Install an old version of pytorch: `conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia`
* CIFAR-10 random forgetting (under `src/` folder): `./cifar10_random_forgetting.sh 0`, where `0` is the GPU id. 
* CIFAR-100 random forgetting (under `src/` folder): `./cifar100_random_forgetting.sh 0`.
