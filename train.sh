#!/bin/bash

set -x

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=3,5,6,7

NNODES=${NNODES:=1}
NPROC_PER_NODE=${NPROC_PER_NODE:=4}
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}



torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT $@ 2>&1 | tee log.txt
