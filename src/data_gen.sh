#!/bin/bash

# Make only GPU 0 and GPU 2 visible
source ~/.bashrc

# Initialize conda for shell use
eval "$(conda shell.bash hook)"

export CUDA_VISIBLE_DEVICES=1

# Your command here
conda activate gsam
./run_gsam.sh animal --max-images 10 --output-dir /data3/jhpark/testing_10
conda activate rf-solver
./run_flux.sh  /data3/jhpark/testing_10 --output-dir /data3/jhpark/testing_10
conda activate gsam
python unified_data_pipeline.py --gsam_dir /data3/jhpark/testing_10 --flux_dir /data3/jhpark/testing_10 --output_dir /data3/jhpark/filtered_testing_10
