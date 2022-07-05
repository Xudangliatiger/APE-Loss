#！/bin/bash
CONFIG=configs/emp/emp_ape_loss_atss_r50_fpn_giou_sigmoid_8_24e_coco1333.py
CHECKPOINT=/mnt/data0/home/xudongli/fap/work_dirs/emp_ape_loss_atss_r50_fpn_giou_sigmoid_8_24e_coco1333/epoch_12.pth
GPUS=4
PORT=$((RANDOM + 10000))
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py --eval bbox --launcher pytorch ${@:4} $CONFIG $CHECKPOINT