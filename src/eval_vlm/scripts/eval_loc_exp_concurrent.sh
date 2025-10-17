#!/bin/bash

# Concurrent Evaluation Script for Localization + Explanation Models
# This script runs multiple evaluations in parallel with different GPU assignments

# Define an array of datasets to evaluate
DATASETS=("ours" "synthscars" "synartifact" "loki" "richhf")

# Define available GPUs (adjust based on your system)
GPUS=(0 1 2 3 4 5 6 7)

# Initialize arrays to track failed experiments and running processes
FAILED_EXPERIMENTS=()
RUNNING_PIDS=()
COMPLETED_EXPERIMENTS=()

# Function to get dataset path
get_dataset_path() {
    local dataset=$1
    case $dataset in
        "synthscars")
            echo "/data2/jhpark/image-artifacts/data/eval/SynthScars/test"
            ;;
        "synartifact")
            echo "/data2/jhpark/image-artifacts/data/eval/SynArtifact"
            ;;
        "loki")
            echo "/data2/jhpark/image-artifacts/data/eval/loki"
            ;;
        "richhf")
            echo "/data2/jhpark/image-artifacts/data/eval/richhf-18k"
            ;;
        *)
            echo "/data2/jhpark/image-artifacts/data/eval"
            ;;
    esac
}

# Function to run a single evaluation
run_evaluation() {
    local gpu_id=$1
    local exp_dir=$2
    local dataset=$3
    local dataset_path=$4
    local model_type=$5
    local experiment_name=$6
    
    echo "[GPU $gpu_id] Starting evaluation: $experiment_name"
    
    # Create log directory
    mkdir -p "eval_logs/concurrent_logs"
    
    # Run the evaluation with logging
    CUDA_VISIBLE_DEVICES=$gpu_id python eval_multi_turn_loc_exp_batch.py \
        --exp-dir "$exp_dir" \
        --dataset "$dataset" \
        --dataset-path "$dataset_path" \
        --device cuda:0 \
        > "eval_logs/concurrent_logs/${experiment_name}.log" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[GPU $gpu_id] ✅ Completed: $experiment_name"
        echo "$experiment_name" >> "eval_logs/concurrent_logs/completed_experiments.txt"
    else
        echo "[GPU $gpu_id] ❌ Failed: $experiment_name (exit code: $exit_code)"
        echo "$experiment_name" >> "eval_logs/concurrent_logs/failed_experiments.txt"
        FAILED_EXPERIMENTS+=("$experiment_name")
    fi
}

# Function to wait for available GPU
wait_for_gpu() {
    while [ ${#RUNNING_PIDS[@]} -ge ${#GPUS[@]} ]; do
        # Check for completed processes
        local new_pids=()
        for pid in "${RUNNING_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        RUNNING_PIDS=("${new_pids[@]}")
        
        if [ ${#RUNNING_PIDS[@]} -ge ${#GPUS[@]} ]; then
            sleep 5
        fi
    done
}

# Function to get next available GPU
get_next_gpu() {
    local gpu_index=$((${#RUNNING_PIDS[@]} % ${#GPUS[@]}))
    echo "${GPUS[$gpu_index]}"
}

# Function to start evaluation in background
start_evaluation() {
    local exp_dir=$1
    local dataset=$2
    local dataset_path=$3
    local model_type=$4
    local experiment_name=$5
    
    wait_for_gpu
    local gpu_id=$(get_next_gpu)
    
    # Start evaluation in background
    run_evaluation "$gpu_id" "$exp_dir" "$dataset" "$dataset_path" "$model_type" "$experiment_name" &
    local pid=$!
    RUNNING_PIDS+=("$pid")
    
    echo "[GPU $gpu_id] Started PID $pid: $experiment_name"
}

# Create log directory
mkdir -p "eval_logs/concurrent_logs"
echo "Starting concurrent evaluation at $(date)" > "eval_logs/concurrent_logs/start_time.txt"

echo "🚀 Starting Concurrent Evaluation"
echo "Available GPUs: ${GPUS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "=========================================="

# Start all evaluations concurrently
echo "📊 Starting Multi-Turn Vanilla Model Evaluations..."

# Multi-turn vanilla model evaluations
for DATASET in "${DATASETS[@]}"; do
    DATASET_PATH=$(get_dataset_path "$DATASET")
    EXPERIMENT_NAME="multi_turn_loc_exp_vanilla_${DATASET}"
    
    start_evaluation \
        "/data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/multi_turn_loc_exp_vanilla" \
        "$DATASET" \
        "$DATASET_PATH" \
        "multi_turn_vanilla" \
        "$EXPERIMENT_NAME"
done

# Multi-turn vanilla validation
start_evaluation \
    "/data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/multi_turn_loc_exp_vanilla" \
    "val" \
    "/home/jhpark/image-artifacts/src/train/LLaMA-Factory/data/multi_turn_loc_exp_val.json" \
    "multi_turn_vanilla" \
    "multi_turn_loc_exp_vanilla_val"

echo "📊 Starting Multi-Turn Fireflow Model Evaluations..."

# Multi-turn fireflow model evaluations
for DATASET in "${DATASETS[@]}"; do
    DATASET_PATH=$(get_dataset_path "$DATASET")
    EXPERIMENT_NAME="multi_turn_loc_exp_fireflow_${DATASET}"
    
    start_evaluation \
        "/data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/1000/multi_turn_loc_exp_fireflow" \
        "$DATASET" \
        "$DATASET_PATH" \
        "multi_turn_fireflow" \
        "$EXPERIMENT_NAME"
done

# Multi-turn fireflow validation
start_evaluation \
    "/data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/1000/multi_turn_loc_exp_fireflow" \
    "val" \
    "/home/jhpark/image-artifacts/src/train/LLaMA-Factory/data/multi_turn_loc_exp_val.json" \
    "multi_turn_fireflow" \
    "multi_turn_loc_exp_fireflow_val"

echo "⏳ Waiting for all evaluations to complete..."

# Wait for all background processes to complete
for pid in "${RUNNING_PIDS[@]}"; do
    wait "$pid"
done

echo "✅ All evaluations completed!"

# Collect results from log files
if [ -f "eval_logs/concurrent_logs/failed_experiments.txt" ]; then
    while IFS= read -r line; do
        FAILED_EXPERIMENTS+=("$line")
    done < "eval_logs/concurrent_logs/failed_experiments.txt"
fi

if [ -f "eval_logs/concurrent_logs/completed_experiments.txt" ]; then
    while IFS= read -r line; do
        COMPLETED_EXPERIMENTS+=("$line")
    done < "eval_logs/concurrent_logs/completed_experiments.txt"
fi

# Print summary
echo ""
echo "=========================================="
echo "CONCURRENT EVALUATION SUMMARY"
echo "=========================================="
echo "Completed at: $(date)"
echo ""

if [ ${#COMPLETED_EXPERIMENTS[@]} -gt 0 ]; then
    echo "✅ Successfully completed experiments (${#COMPLETED_EXPERIMENTS[@]}):"
    for experiment in "${COMPLETED_EXPERIMENTS[@]}"; do
        echo "  - $experiment"
    done
    echo ""
fi

if [ ${#FAILED_EXPERIMENTS[@]} -eq 0 ]; then
    echo "🎉 All experiments completed successfully!"
else
    echo "❌ Failed experiments (${#FAILED_EXPERIMENTS[@]}):"
    for experiment in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  - $experiment"
    done
    echo ""
    echo "Check individual log files in eval_logs/concurrent_logs/ for details."
fi

echo ""
echo "📁 Log files saved in: eval_logs/concurrent_logs/"
echo "=========================================="
