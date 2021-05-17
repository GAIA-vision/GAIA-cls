#!/usr/bin/env bash

CONFIG=$1
CKPT_PATH=$2
GPUS=$3
MODEL_SPACE_PATH=$5
PORT=${PORT:-29500}


PYTHONPATH=/path/to/gaiavision:"$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$(dirname $0)/../tools"/test_supernet.py \
    ${CONFIG} \
    ${CKPT_PATH} \
    --work-dir ${WORK_DIR} \
    --launcher pytorch \
    --eval bbox
