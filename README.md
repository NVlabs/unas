# UNAS: Differentiable Architecture Search Meets Reinforcement Learning, CVPR 2020 Oral [(Paper)](https://arxiv.org/abs/1912.07651)

<div align="center">
  <a href="http://latentspace.cc/arash_vahdat/" target="_blank">Arash&nbsp;Vahdat</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://arunmallya.github.io/" target="_blank">Arun&nbsp;Mallya</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://mingyuliu.net/" target="_blank">Ming&#8209;Yu&nbsp;Liu</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://jankautz.com/" target="_blank">Jan&nbsp;Kautz</a> 
</div>

## Introduction

This repository provides the official PyTorch implementation of UNAS that was presented at CVPR 2020. 
The paper presents results in two search spaces including [DARTS](https://arxiv.org/abs/1806.09055) 
and [ProxylessNAS](https://arxiv.org/abs/1812.00332) spaces. Our paper
can be found [here](https://arxiv.org/abs/1912.07651).


### Watch a short video that describes UNAS:
<div align="center">
<a href=https://www.youtube.com/watch?v=UZboUDcGL70><img src="https://img.youtube.com/vi/UZboUDcGL70/0.jpg" width="500"> </a> 
</div>

## Requirements
UNAS is examined using Python 3.5. To install the requirements, follow these steps:
1. Run 
```
$ pip install -r requirements.txt
```
2. Install [apex](https://github.com/NVIDIA/apex) using:
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir ./
```


## Preparing Data
We examined UNAS on the CIFAR-10, CIFAR-100, and ImageNet datasets. The CIFAR datasets will be downloaded
automatically if you don't have a local copy. However, you may need to download the ImageNet 2012 dataset 
and structure the dataset under  train and val subfloders. You can 
follow [this page](https://github.com/pytorch/examples/tree/master/imagenet#requirements) 
to structure the dataset. The data directory should be in the form:

    data/
        ├── train/
            ├── n01440764/
            ├── n01443537/
            ├── ...
        ├── val/
            ├── n01440764/
            ├── n01443537/
            ├── ...        
 

## Running Search and Evaluation from Scratch
In UNAS, in the search phase, we first search for either normal/reduction cells (in DARTS space) or the whole network architecture
(in ProxylessNAS space). Then in the evaluation phase, in order to measure the performance of search, 
we train the discovered cells or network architectures from scratch. The commands for performing
both search and evaluation are provided in `commands.sh`. Note that we use 4 GPUs for search on CIFAR-10
and CIFAR-100, 1 GPU for evaluation on these datasets, and 8 GPUs for search and eval on ImageNet.

## Running with Cells/Networks Discovered by UNAS
In order to run the evaluation code with the cells discovered in our experiments, you can set `--genotype` to
`UNAS_CIFAR10`, `UNAS_CIFAR100`, or `UNAS_IMAGENET` when evaluating in DARTS space. Similarly, you can
set `--genotype` to `UNAS` when evaluating in ProxylessNAS space (check the example command in `commands.sh`.

## Monitoring search/eval progress
You can monitor search and evaluation progress using tensorboar. 
You can launch tensorboard using these commands:
```
$ export EXPR_ID=UNIQUE_EXPR_ID
$ export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
$ tensorboard --logdir $CHECKPOINT_DIR/search-$EXPR_ID/       # During search
$ tensorboard --logdir $CHECKPOINT_DIR/eval-$EXPR_ID/         # During evaluation
```

### License
Please check the LICENSE file.

### Bibtex:
```
@inproceedings{vahdat2019UNAS,
  title={{UNAS}: Differentiable Architecture Search Meets Reinforcement Learning},
  author={Vahdat, Arash and Mallya, Arun and Liu, Ming-Yu and  Kautz, Jan},
  booktitle = {Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```
