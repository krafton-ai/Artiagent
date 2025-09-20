#!/bin/bash

# Make only GPU 0 and GPU 2 visible

source ~/.bashrc

# Initialize conda for shell use
eval "$(conda shell.bash hook)"

parent_dir="/data3/jhpark/image-artifact-real-images"
data_dir="coco/scene"
output_dir="/data3/jhpark/data_synth"
filtered_output_dir="/data3/jhpark/filtered_data_synth"


export CUDA_VISIBLE_DEVICES=3
# Your command here
# conda activate gsam
# ./run_gsam.sh custom --dataset custom --dataset-path $parent_dir/$data_dir --output-dir $output_dir/$data_dir
conda activate rf-solver
./run_flux.sh  $output_dir/$data_dir --output-dir $output_dir/$data_dir
conda activate gsam
python unified_data_pipeline.py --gsam_dir $output_dir/$data_dir --flux_dir $output_dir/$data_dir --output_dir $filtered_output_dir/$data_dir
# cd /data3/jhpark