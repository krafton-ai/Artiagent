#!/bin/bash

# Make only GPU 0 and GPU 2 visible

source ~/.bashrc

# Initialize conda for shell use
eval "$(conda shell.bash hook)"

parent_dir="/data3/jhpark/image-artifact-real-images"
data_dir="coco_testing/scene"
output_dir="/data3/jhpark/data_synth"
filtered_output_dir="/data3/jhpark/filtered_data_synth"
chunk_name="chunk_01"


export CUDA_VISIBLE_DEVICES=2
# Your command here
conda activate gsam
./run_gsam.sh custom --dataset custom --dataset-path $parent_dir/$data_dir/$chunk_name --max-images 10 --output-dir $output_dir/$data_dir/$chunk_name
conda activate rf-solver
./run_flux.sh  $output_dir/$data_dir/$chunk_name --output-dir $output_dir/$data_dir/$chunk_name
conda activate gsam
python unified_data_pipeline.py --gsam_dir $output_dir/$data_dir/$chunk_name --flux_dir $output_dir/$data_dir/$chunk_name --output_dir $filtered_output_dir/$data_dir/$chunk_name
cd /data3/jhpark
zip -r $filtered_output_dir/$data_dir/$chunk_name.zip $filtered_output_dir/$data_dir/$chunk_name
