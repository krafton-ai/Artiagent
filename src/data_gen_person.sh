#!/bin/bash

# Make only GPU 0 and GPU 2 visible

source ~/.bashrc

# Initialize conda for shell use
eval "$(conda shell.bash hook)"

export CUDA_VISIBLE_DEVICES=3
# Your command here
conda activate gsam
./run_gsam.sh person --max-images 1000 --output-dir /data3/jhpark/person_1k
conda activate rf-solver
./run_flux.sh  /data3/jhpark/person_1k --output-dir /data3/jhpark/person_1k
conda activate gsam
python unified_data_pipeline.py --gsam_dir /data3/jhpark/person_1k --flux_dir /data3/jhpark/person_1k --output_dir /data3/jhpark/filtered_person_1k
cd /data3/jhpark
zip -r filtered_person_1k.zip filtered_person_1k
