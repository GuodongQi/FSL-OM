# Transductive Few-Shot Classification on the Oblique Manifold

## Introduction
The folders contain the code for  paper [Transductive Few-Shot Classification on the Oblique Manifold](https://arxiv.org/abs/2108.04009).


## Environment

    numpy==1.18.5
    torch==1.7.0
    Pillow==7.2.0
    torchvision==0.8.0
    tqdm==4.46.0
    
## Train

For example:\
`5-way 5-shot` with `resnet18` in `mini-ImageNet`:\
``python train.py --n_way 5 --k_shot 5 --k_query 15 --skip False --dataset mini --backbone resnet18 --gpu 0,1,2``\

`5-way 1-shot` with `WRN` in `tiered-ImageNet`:\
``python train.py --n_way 5 --k_shot 1 --k_query 15 --skip False --dataset tiered --backbone wideres --gpu 0,1,2``\

## Use the Pretrained Models

Move the  models to folder ``checkpoint``, and change the argument `skip`  with value `True`:\
For example:\
`5-way 5-shot` with `resnet18` in `mini-ImageNet`:\
``python train.py --n_way 5 --k_shot 5 --k_query 15 --skip True --dataset mini --backbone resnet18 --gpu 0,1,2``\

`5-way 1-shot` with `WRN` in `tiered-ImageNet`:\
``python train.py --n_way 5 --k_shot 1 --k_query 15 --skip True --dataset tiered --backbone wideres --gpu 0,1,2``\

## Reference
* [TIM](https://github.com/mboudiaf/TIM)
* [Geoopt](https://github.com/geoopt/geoopt)
