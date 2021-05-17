#!/usr/bin/env bash

GPUS=$2
CONFIG=$1
WORK_DIR=$3
PORT=${PORT:-29500}


PYTHONPATH=/path/to/gaiavision:"$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$(dirname $0)/../tools"/count_flops.py \
    ${CONFIG} \
    --launcher pytorch 

