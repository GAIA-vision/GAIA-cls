#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH=/mnt/diske/qing_chang/GAIA/GAIA-cv-dev:"$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$(dirname $0)/../tools"/train_supernet.py \
    ${CONFIG} \
    --launcher pytorch