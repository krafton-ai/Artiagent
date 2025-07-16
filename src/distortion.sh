#!/bin/bash

# Make only GPU 0 and GPU 2 visible
source ~/.bashrc

# Initialize conda for shell use
eval "$(conda shell.bash hook)"

export CUDA_VISIBLE_DEVICES=0

# Your command here
conda activate gsam
./run_gsam.sh animal --max-images 200 --output-dir ../exps/1k/distortion --artifact-types "distortion" --random-distortion
conda activate rf-solver
./run_flux.sh  ../exps/1k/distortion --output-dir ../exps/1k/distortion
conda activate gsam
python data_filter_pipeline.py --gsam_dir ../exps/1k/distortion --flux_dir ../exps/1k/distortion --output_dir ../exps/filtering/distortion