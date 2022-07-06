#!/usr/bin/env bash
#！/bin/bash
CONFIG=configs/ape_loss/ape_loss_paa_r50_fpn_giou_sigmoid_8_24e_coco1333.py
GPUS=4
PORT=$((RANDOM + 10000))
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}