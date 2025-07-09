#!/bin/bash

# Make only GPU 0 and GPU 2 visible
source ~/.bashrc

# Initialize conda for shell use
eval "$(conda shell.bash hook)"

export CUDA_VISIBLE_DEVICES=1

# Your command here
conda activate gsam
./run_gsam.sh animal --artifact-types "addition" --output-dir ../exps/1k/addition --max-images 20
conda activate rf-solver
./run_flux.sh  ../exps/1k/addition --output-dir ../exps/1k/addition
conda activate gsam
python data_filter_pipeline.py --gsam_dir ../exps/1k/addition --flux_dir ../exps/1k/addition --output_dir ../exps/filtering/addition