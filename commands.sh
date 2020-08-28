#!/bin/bash

##################################
####### DARTS Search Space #######
##################################

########## Search on CIFAR100 #############
# use a unique id for each search.
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
cd ./darts
python -m torch.distributed.launch --nproc_per_node=4 train_search.py --data $DATA_DIR \
--root $CHECKPOINT_DIR --save $EXPR_ID --seed 0 --arch_learning_rate 1e-3 --meta_loss rebar \
--gen_error_alpha  --gen_error_alpha_lambda 0.5 --dataset cifar10 --scale_lr

## Evaluate discovered cell on CIFAR10 by training the cell from scratch ###
cd ./darts
python -m torch.distributed.launch --nproc_per_node=1  --master_port=8000 train_eval_cifar.py \
--data $DATA_DIR --root_dir $CHECKPOINT_DIR --save $EXPR_ID \
--genotype $CHECKPOINT_DIR/search-$EXPR_ID/genotype.pt --dataset cifar10  \
--batch_size 128 --warmup_epochs 30 --learning_rate 0.05 --drop_path_prob 0.2


########## Search on CIFAR100 #############
# use a unique id for each search.
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
cd ./darts
python -m torch.distributed.launch --nproc_per_node=4 train_search.py --data $DATA_DIR \
--root $CHECKPOINT_DIR --save $EXPR_ID --seed 0 --arch_learning_rate 1e-3 --meta_loss rebar \
--gen_error_alpha  --gen_error_alpha_lambda 0.5 --dataset cifar100 --scale_lr

## Evaluate discovered cell on CIFAR100 by training the cell from scratch ###
cd ./darts
python -m torch.distributed.launch --nproc_per_node=1  --master_port=8000 train_eval_cifar.py \
--data $DATA_DIR --root_dir $CHECKPOINT_DIR --save UNIQUE_EXPR_ID \
--genotype $CHECKPOINT_DIR/search-$EXPR_ID/genotype.pt --dataset cifar100  --batch_size 128 \
--warmup_epochs 30 --learning_rate 0.05 --drop_path_prob 0.3

########## Search on ImageNet #############
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_IMAGENET_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
cd ./darts
python -m torch.distributed.launch --nproc_per_node=8 train_search.py --data $DATA_DIR \
--root $CHECKPOINT_DIR --save $EXPR_ID --layers 14 --seed 0 --num_ops 5 --arch_learning_rate 1e-3 \
--gsm_soften_eps 0.08 --meta_loss rebar --gen_error_alpha  --gen_error_alpha_lambda 0.5 --init_epoch 5 \
--epochs 15 --batch_size 24 --dataset imagenet --same_alpha_minibatch

## Evaluate discovered cell on ImageNet by training the cell from scratch ###
cd ./darts
python -m torch.distributed.launch --nproc_per_node=8 train_eval_imagenet.py --data $DATA_DIR \
--root_dir $CHECKPOINT_DIR --save $EXPR_ID --genotype $CHECKPOINT_DIR/search-$EXPR_ID/genotype.pt


###########################################################
####### ProxylessNAS Search Space with Latency Loss #######
###########################################################

########## Search on ImageNet targeting 10 ms latency in PyTorch #############
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_IMAGENET_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
cd ./proxylessnas
python -m torch.distributed.launch --nproc_per_node=8 train_search.py --data $DATA_DIR \
--root $CHECKPOINT_DIR --save $EXPR_ID --seed 0 --arch_learning_rate 1e-3 --learning_rate 3e-4 \
--meta_loss rebar --gen_error_alpha  --gen_error_alpha_lambda 0.5 --init_epoch 20 --epochs 15 \
--batch_size 24 --dataset imagenet --target_latency 10. --same_alpha_minibatch

## Evaluate discovered network on ImageNet by training it from scratch ###
cd ./proxylessnas
python -m torch.distributed.launch --nproc_per_node=8 train_eval.py --data $DATA_DIR \
--root_dir $CHECKPOINT_DIR --save $EXPR_ID --genotype $CHECKPOINT_DIR/search-$EXPR_ID/genotype.pt


## Example command: evaluate our discovered network in the proxylessnas space ##
cd ./proxylessnas
python -m torch.distributed.launch --nproc_per_node=8 train_eval.py --data $DATA_DIR \
--root_dir $CHECKPOINT_DIR --save $EXPR_ID --genotype UNAS

