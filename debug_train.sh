#!/bin/bash

set -x

export TOKENIZERS_PARALLELISM=false

# Use 2 GPUs but only show one to force DDP
export CUDA_VISIBLE_DEVICES=3,5,6,7
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12345

# This will force DDP initialization
torchrun --nnodes=1 --nproc-per-node=4 --node-rank=0 \
  --master-addr=localhost --master-port=12345 $@ 2>&1 | tee debug_log.txt