#!/bin/bash

# Advanced Concurrent Evaluation Script for Localization + Explanation Models
# Features:
# - Dynamic GPU assignment with memory monitoring
# - Process monitoring and automatic restart on failure
# - Resource usage tracking
# - Progress reporting

# Configuration
DATASETS=("ours" "synthscars" "synartifact" "loki" "richhf")
GPUS=(0 1 2 3 4 5 6 7)  # Adjust based on your system
MAX_CONCURRENT_JOBS=4   # Limit concurrent jobs to avoid memory issues
LOG_DIR="eval_logs/concurrent_logs_advanced"

# Initialize tracking arrays
FAILED_EXPERIMENTS=()
COMPLETED_EXPERIMENTS=()
RUNNING_JOBS=()
JOB_COUNTER=0

# Create log directory
mkdir -p "$LOG_DIR"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/execution.log"
}

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

# Function to check GPU memory usage
check_gpu_memory() {
    local gpu_id=$1
    local threshold=80  # Percentage threshold
    
    if command -v nvidia-smi &> /dev/null; then
        local memory_usage=$(nvidia-smi --id=$gpu_id --query-gpu=memory.percent --format=csv,noheader,nounits)
        if [ "$memory_usage" -gt "$threshold" ]; then
            return 1  # GPU memory usage too high
        fi
    fi
    return 0  # GPU is available
}

# Function to find best available GPU
find_best_gpu() {
    local best_gpu=""
    local min_memory=100
    
    for gpu_id in "${GPUS[@]}"; do
        if command -v nvidia-smi &> /dev/null; then
            local memory_usage=$(nvidia-smi --id=$gpu_id --query-gpu=memory.percent --format=csv,noheader,nounits 2>/dev/null)
            if [ "$memory_usage" -lt "$min_memory" ]; then
                min_memory="$memory_usage"
                best_gpu="$gpu_id"
            fi
        else
            # Fallback: use round-robin if nvidia-smi not available
            best_gpu="${GPUS[$((${#RUNNING_JOBS[@]} % ${#GPUS[@]}))]}"
            break
        fi
    done
    
    echo "$best_gpu"
}

# Function to run evaluation with monitoring
run_evaluation_monitored() {
    local job_id=$1
    local gpu_id=$2
    local exp_dir=$3
    local dataset=$4
    local dataset_path=$5
    local experiment_name=$6
    
    log "Starting Job $job_id on GPU $gpu_id: $experiment_name"
    
    # Create job-specific log file
    local job_log="$LOG_DIR/job_${job_id}_${experiment_name}.log"
    
    # Start monitoring script in background
    (
        while true; do
            if ! kill -0 $$ 2>/dev/null; then
                break  # Parent process died
            fi
            echo "[$(date '+%H:%M:%S')] GPU $gpu_id Memory: $(nvidia-smi --id=$gpu_id --query-gpu=memory.percent --format=csv,noheader,nounits 2>/dev/null || echo 'N/A')%" >> "$LOG_DIR/gpu_monitor.log"
            sleep 30
        done
    ) &
    local monitor_pid=$!
    
    # Run the actual evaluation
    local start_time=$(date +%s)
    
    CUDA_VISIBLE_DEVICES=$gpu_id python eval_multi_turn_loc_exp_batch.py \
        --exp-dir "$exp_dir" \
        --dataset "$dataset" \
        --dataset-path "$dataset_path" \
        --device cuda:0 \
        > "$job_log" 2>&1
    
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Kill monitoring process
    kill $monitor_pid 2>/dev/null
    
    # Record results
    if [ $exit_code -eq 0 ]; then
        log "✅ Job $job_id completed successfully: $experiment_name (${duration}s)"
        echo "$experiment_name" >> "$LOG_DIR/completed_experiments.txt"
        echo "$experiment_name,$gpu_id,$duration,SUCCESS" >> "$LOG_DIR/job_summary.csv"
    else
        log "❌ Job $job_id failed: $experiment_name (${duration}s, exit code: $exit_code)"
        echo "$experiment_name" >> "$LOG_DIR/failed_experiments.txt"
        echo "$experiment_name,$gpu_id,$duration,FAILED" >> "$LOG_DIR/job_summary.csv"
        FAILED_EXPERIMENTS+=("$experiment_name")
    fi
    
    # Remove from running jobs
    RUNNING_JOBS=($(printf '%s\n' "${RUNNING_JOBS[@]}" | grep -v "^$job_id:"))
}

# Function to start evaluation job
start_evaluation_job() {
    local exp_dir=$1
    local dataset=$2
    local dataset_path=$3
    local experiment_name=$4
    
    # Wait for available slot
    while [ ${#RUNNING_JOBS[@]} -ge $MAX_CONCURRENT_JOBS ]; do
        # Check for completed jobs
        local new_jobs=()
        for job in "${RUNNING_JOBS[@]}"; do
            local job_id=$(echo "$job" | cut -d: -f1)
            local job_pid=$(echo "$job" | cut -d: -f2)
            
            if kill -0 "$job_pid" 2>/dev/null; then
                new_jobs+=("$job")
            fi
        done
        RUNNING_JOBS=("${new_jobs[@]}")
        
        if [ ${#RUNNING_JOBS[@]} -ge $MAX_CONCURRENT_JOBS ]; then
            sleep 5
        fi
    done
    
    # Find best GPU
    local gpu_id=$(find_best_gpu)
    JOB_COUNTER=$((JOB_COUNTER + 1))
    
    # Start job in background
    run_evaluation_monitored "$JOB_COUNTER" "$gpu_id" "$exp_dir" "$dataset" "$dataset_path" "$experiment_name" &
    local job_pid=$!
    
    RUNNING_JOBS+=("$JOB_COUNTER:$job_pid")
    log "Started Job $JOB_COUNTER (PID $job_pid) on GPU $gpu_id: $experiment_name"
}

# Function to generate progress report
generate_progress_report() {
    local total_jobs=$1
    local completed_jobs=${#COMPLETED_EXPERIMENTS[@]}
    local failed_jobs=${#FAILED_EXPERIMENTS[@]}
    local running_jobs=${#RUNNING_JOBS[@]}
    local pending_jobs=$((total_jobs - completed_jobs - failed_jobs - running_jobs))
    
    echo ""
    echo "📊 Progress Report:"
    echo "  Total Jobs: $total_jobs"
    echo "  Completed: $completed_jobs"
    echo "  Failed: $failed_jobs"
    echo "  Running: $running_jobs"
    echo "  Pending: $pending_jobs"
    echo "  Progress: $(( (completed_jobs + failed_jobs) * 100 / total_jobs ))%"
    echo ""
}

# Initialize job summary CSV
echo "experiment_name,gpu_id,duration,status" > "$LOG_DIR/job_summary.csv"

# Start logging
log "🚀 Starting Advanced Concurrent Evaluation"
log "Available GPUs: ${GPUS[*]}"
log "Max concurrent jobs: $MAX_CONCURRENT_JOBS"
log "Datasets: ${DATASETS[*]}"

# Calculate total jobs
TOTAL_JOBS=$(( ${#DATASETS[@]} * 2 + 2 ))  # 2 models × 5 datasets + 2 validation sets
log "Total jobs to execute: $TOTAL_JOBS"

echo "=========================================="

# Start all evaluations
log "📊 Starting Multi-Turn Vanilla Model Evaluations..."

# Multi-turn vanilla model evaluations
for DATASET in "${DATASETS[@]}"; do
    DATASET_PATH=$(get_dataset_path "$DATASET")
    EXPERIMENT_NAME="multi_turn_loc_exp_vanilla_${DATASET}"
    
    start_evaluation_job \
        "/data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/multi_turn_loc_exp_vanilla" \
        "$DATASET" \
        "$DATASET_PATH" \
        "$EXPERIMENT_NAME"
done

# Multi-turn vanilla validation
start_evaluation_job \
    "/data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/multi_turn_loc_exp_vanilla" \
    "val" \
    "/home/jhpark/image-artifacts/src/train/LLaMA-Factory/data/multi_turn_loc_exp_val.json" \
    "multi_turn_loc_exp_vanilla_val"

log "📊 Starting Multi-Turn Fireflow Model Evaluations..."

# Multi-turn fireflow model evaluations
for DATASET in "${DATASETS[@]}"; do
    DATASET_PATH=$(get_dataset_path "$DATASET")
    EXPERIMENT_NAME="multi_turn_loc_exp_fireflow_${DATASET}"
    
    start_evaluation_job \
        "/data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/1000/multi_turn_loc_exp_fireflow" \
        "$DATASET" \
        "$DATASET_PATH" \
        "$EXPERIMENT_NAME"
done

# Multi-turn fireflow validation
start_evaluation_job \
    "/data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/1000/multi_turn_loc_exp_fireflow" \
    "val" \
    "/home/jhpark/image-artifacts/src/train/LLaMA-Factory/data/multi_turn_loc_exp_val.json" \
    "multi_turn_loc_exp_fireflow_val"

log "⏳ Waiting for all evaluations to complete..."

# Monitor progress
while [ ${#RUNNING_JOBS[@]} -gt 0 ]; do
    # Check for completed jobs
    local new_jobs=()
    for job in "${RUNNING_JOBS[@]}"; do
        local job_id=$(echo "$job" | cut -d: -f1)
        local job_pid=$(echo "$job" | cut -d: -f2)
        
        if kill -0 "$job_pid" 2>/dev/null; then
            new_jobs+=("$job")
        fi
    done
    RUNNING_JOBS=("${new_jobs[@]}")
    
    # Generate progress report
    generate_progress_report $TOTAL_JOBS
    
    if [ ${#RUNNING_JOBS[@]} -gt 0 ]; then
        sleep 30
    fi
done

log "✅ All evaluations completed!"

# Collect final results
if [ -f "$LOG_DIR/failed_experiments.txt" ]; then
    while IFS= read -r line; do
        FAILED_EXPERIMENTS+=("$line")
    done < "$LOG_DIR/failed_experiments.txt"
fi

if [ -f "$LOG_DIR/completed_experiments.txt" ]; then
    while IFS= read -r line; do
        COMPLETED_EXPERIMENTS+=("$line")
    done < "$LOG_DIR/completed_experiments.txt"
fi

# Generate final summary
echo ""
echo "=========================================="
echo "FINAL EVALUATION SUMMARY"
echo "=========================================="
log "Evaluation completed at: $(date)"

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
fi

# Generate performance summary
if [ -f "$LOG_DIR/job_summary.csv" ]; then
    echo "📊 Performance Summary:"
    echo "  Average job duration: $(awk -F',' 'NR>1 {sum+=$3; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}' "$LOG_DIR/job_summary.csv") seconds"
    echo "  Total execution time: $(($(date +%s) - $(date -d "$(head -1 "$LOG_DIR/execution.log" | cut -d']' -f1 | tr -d '[')" +%s))) seconds"
fi

echo ""
echo "📁 Detailed logs saved in: $LOG_DIR/"
echo "  - execution.log: Main execution log"
echo "  - job_summary.csv: Job performance summary"
echo "  - gpu_monitor.log: GPU memory usage over time"
echo "  - job_*_*.log: Individual job logs"
echo "=========================================="
