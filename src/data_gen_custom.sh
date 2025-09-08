#!/bin/bash

# Make only GPU 0 and GPU 2 visible

source ~/.bashrc

# Initialize conda for shell use
eval "$(conda shell.bash hook)"

export CUDA_VISIBLE_DEVICES=1
# Your command here
conda activate gsam
./run_gsam.sh custom --dataset custom --dataset-path /data3/jhpark/image-artifact-real-images/caltech/batch_001 --max-images 10 --output-dir /data3/jhpark/new_testing_custom
conda activate rf-solver
./run_flux.sh  /data3/jhpark/new_testing_custom --output-dir /data3/jhpark/new_testing_custom
conda activate gsam
python unified_data_pipeline.py --gsam_dir /data3/jhpark/new_testing_custom --flux_dir /data3/jhpark/new_testing_custom --output_dir /data3/jhpark/filtered_new_testing_custom
cd /data3/jhpark
zip -r filtered_new_testing_custom.zip filtered_new_testing_custom
