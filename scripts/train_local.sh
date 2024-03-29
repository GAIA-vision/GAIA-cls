#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-21701}
PYTHONPATH=../gaiavision:"$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    ./tools/train_supernet.py \
    ${CONFIG} \
    --launcher pytorch


