#!/bin/bash

# Make only GPU 0 and GPU 2 visible
source ~/.bashrc

# Initialize conda for shell use
eval "$(conda shell.bash hook)"

export CUDA_VISIBLE_DEVICES=0

# Your command here
conda activate gsam
./run_gsam.sh animal --max-images 1000 --output-dir /data3/jhpark/artifacts_1k
conda activate rf-solver
./run_flux.sh  /data3/jhpark/artifacts_1k --output-dir /data3/jhpark/artifacts_1k
