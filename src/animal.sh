#!/bin/bash

# Make only GPU 0 and GPU 2 visible
export CUDA_VISIBLE_DEVICES=1

# Your command here
./run_vlpart.sh animal --max-images 50 --output-dir ../exps/vlpart_animal_new
./run_flux.sh  ../exps/vlpart_animal_new --output-dir ../exps/vlpart_animal_distort_flux