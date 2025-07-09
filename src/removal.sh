#!/bin/bash

# Make only GPU 0 and GPU 2 visible
source ~/.bashrc

# Initialize conda for shell use
eval "$(conda shell.bash hook)"

export CUDA_VISIBLE_DEVICES=0

# Your command here
conda activate gsam
./run_gsam.sh animal --artifact-types "removal" --output-dir ../exps/1k/removal --max-images 20
conda activate rf-solver
./run_flux.sh  ../exps/1k/removal --output-dir ../exps/1k/removal
conda activate gsam
python data_filter_pipeline.py --gsam_dir ../exps/1k/removal --flux_dir ../exps/1k/removal --output_dir ../exps/filtering/removal