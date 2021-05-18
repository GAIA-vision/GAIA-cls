#!/usr/bin/env bash

CONFIG=$1
WORK_DIR=$2
LOAD_FROM=$3
MODEL_SPACE_PATH=$4
CKPT_PATH=$5
GPUS=$6

PORT=${PORT:-29500}



PYTHONPATH=/mnt/diske/qing_chang/GAIA/GAIA-cls:"$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$(dirname $0)/../tools"/finetune_supernet.py \
    ${CONFIG} \
    --launcher pytorch \
    --work-dir ${WORK_DIR} \
    --eval bbox \
    --load-from ${CKPT_PATH}
