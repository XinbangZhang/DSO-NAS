# DSO-NAS
Codes for papers "You Only Search Once: Single Shot Neural Architecture Search via Direct Sparse Optimization"

> [You Only Search Once: Single Shot Neural Architecture Search via Direct Sparse Optimization](https://arxiv.org/abs/1811.01567)\
> Xinbang zhang, Zehao Huang, Naiyan Wang.

## Requirements
```
python == 2.7 
mxnet == 1.2.1
```
## Cifar Experiments:

Please follow the instructions in https://mxnet.incubator.apache.org/ to make .rec file with the .lst file in  /data/cifar, split  training data into two part (train1.rec & train2.rec) and put them in /data/cifar_search.

Step1:  go to config/cfgs, set dataset='cifar10_search', train_baseline=True, retrain_model=False, model_search=False and run 
```
python train.py
```
Step2:  go to config/cfgs, set dataset='cifar10_search', train_baseline=False,retrain_model=False,model_search=True. To apply share search space, set share_search_space to True. To apply Adaptive Flops or Adaptive MAC technique,  please set ad_flops or ad_mac to True. Run
```
python train_search.py
```
Loading the pretrain model obtained by step1 is recommonded.

Step3:  go to config/cfgs, set dataset='cifar10', train_baseline=False,retrain_model=True,model_search=False, and load the model obtained by step 2 and run 
```
python train.py
```
## ImageNet Experiments:

Please follow the instructions in https://mxnet.incubator.apache.org/ to make .rec file with the .lst file in  /data/imagenet and /data/imagenet_search

Step1:  go to config/cfgs, set dataset='imagenet_search', train_baseline=True,retrain_model=False,model_search=False and run 
```
python train.py
```
Step2:  go to config/cfgs, set dataset='imagenet_search', train_baseline=False,retrain_model=False,model_search=True. To apply share search space, set share_search_space to True. To apply Adaptive Flops or Adaptive Latency technique,  please set ad_flops ad_latency  to True. Run
```
python train_search.py
```
Loading the pretrain model obtained by step1 is recommonded.

Step3:  go to config/cfgs, set dataset='imagenet_search', train_baseline=False,retrain_model=True,model_search=False, and load the model obtained by step 2 and run 
```
python train.py
```
## Bibtex
```
@article{zhang2018you,
  title={You only search once: Single shot neural architecture search via direct sparse optimization},
  author={Zhang, Xinbang and Huang, Zehao and Wang, Naiyan},
  journal={arXiv preprint arXiv:1811.01567},
  year={2018}
}
```
