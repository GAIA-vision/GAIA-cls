#!/usr/bin/env bash

CONFIG=$1
WORK_DIR=$2
SRC_CKPT=$3
GPUS=$4
PORT=${PORT:-29500}


PYTHONPATH=/mnt/diske/qing_chang/GAIA/GAIA-cls:"$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$(dirname $0)/../tools"/extract_subnet.py \
    ${SRC_CKPT} \
    ${WORK_DIR} \
    ${CONFIG} \
    --launcher pytorch

