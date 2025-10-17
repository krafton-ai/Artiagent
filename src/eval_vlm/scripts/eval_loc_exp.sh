
# Define an array of datasets to evaluate
DATASETS=("ours" "synthscars" "synartifact" "loki" "richhf")
# DATASETS=("loki")
# Initialize arrays to track failed experiments
FAILED_EXPERIMENTS=()

# Loop over datasets and run evaluation
# for DATASET in "${DATASETS[@]}"
# do
#     echo "Evaluating dataset: $DATASET"
    
#     # Set dataset-specific path
#     case $DATASET in
#         "synthscars")
#             DATASET_PATH="/data2/jhpark/image-artifacts/data/eval/SynthScars/test"
#             ;;
#         "synartifact")
#             DATASET_PATH="/data2/jhpark/image-artifacts/data/eval/SynArtifact"
#             ;;
#         "loki")
#             DATASET_PATH="/data2/jhpark/image-artifacts/data/eval/loki"
#             ;;
#         "richhf")
#             DATASET_PATH="/data2/jhpark/image-artifacts/data/eval/richhf-18k"
#             ;;
#         *)
#             DATASET_PATH="/data2/jhpark/image-artifacts/data/eval"
#             ;;
#     esac
    
#     CUDA_VISIBLE_DEVICES=5 python eval_single_turn_loc_exp_batch.py \
#         --exp-dir /data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/single_turn_loc_exp_vanilla \
#         --dataset $DATASET \
#         --dataset-path $DATASET_PATH \
#         --device cuda:0 \
#         --batch-size 16 \
        
#     if [ $? -ne 0 ]; then
#         FAILED_EXPERIMENTS+=("single_turn_loc_exp_vanilla_$DATASET")
#     fi
# done

# CUDA_VISIBLE_DEVICES=5 python eval_single_turn_loc_exp_batch.py --exp-dir /data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/single_turn_loc_exp_vanilla --dataset val --dataset-path /home/jhpark/image-artifacts/src/train/LLaMA-Factory/data/single_turn_loc_exp_val.json  
# if [ $? -ne 0 ]; then
#     FAILED_EXPERIMENTS+=("single_turn_loc_exp_vanilla_val")
# fi


# for DATASET in "${DATASETS[@]}"
# do
#     echo "Evaluating dataset: $DATASET"
    
#     # Set dataset-specific path
#     case $DATASET in
#         "synthscars")
#             DATASET_PATH="/data2/jhpark/image-artifacts/data/eval/SynthScars/test"
#             ;;
#         "synartifact")
#             DATASET_PATH="/data2/jhpark/image-artifacts/data/eval/SynArtifact"
#             ;;
#         "loki")
#             DATASET_PATH="/data2/jhpark/image-artifacts/data/eval/loki"
#             ;;
#         "richhf")
#             DATASET_PATH="/data2/jhpark/image-artifacts/data/eval/richhf-18k"
#             ;;
#         *)
#             DATASET_PATH="/data2/jhpark/image-artifacts/data/eval"
#             ;;
#     esac
    
#     CUDA_VISIBLE_DEVICES=5 python eval_single_turn_loc_exp_batch.py \
#         --exp-dir /data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/single_turn_loc_exp_fireflow \
#         --dataset $DATASET \
#         --dataset-path $DATASET_PATH \
#         --device cuda:0 \
#         --batch-size 16 \
        
#     if [ $? -ne 0 ]; then
#         FAILED_EXPERIMENTS+=("single_turn_loc_exp_fireflow_$DATASET")
#     fi
# done

# CUDA_VISIBLE_DEVICES=5 python eval_single_turn_loc_exp_batch.py --exp-dir /data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/single_turn_loc_exp_fireflow --dataset val --dataset-path /home/jhpark/image-artifacts/src/train/LLaMA-Factory/data/single_turn_loc_exp_fireflow_val.json  
# if [ $? -ne 0 ]; then
#     FAILED_EXPERIMENTS+=("single_turn_loc_exp_fireflow_val")
# fi

for DATASET in "${DATASETS[@]}"
do
    echo "Evaluating dataset: $DATASET"
    
    # Set dataset-specific path
    case $DATASET in
        "synthscars")
            DATASET_PATH="/data2/jhpark/image-artifacts/data/eval/SynthScars/test"
            ;;
        "synartifact")
            DATASET_PATH="/data2/jhpark/image-artifacts/data/eval/SynArtifact"
            ;;
        "loki")
            DATASET_PATH="/data2/jhpark/image-artifacts/data/eval/loki"
            ;;
        "richhf")
            DATASET_PATH="/data2/jhpark/image-artifacts/data/eval/richhf-18k"
            ;;
        *)
            DATASET_PATH="/data2/jhpark/image-artifacts/data/eval"
            ;;
    esac
    
    CUDA_VISIBLE_DEVICES=5 python eval_multi_turn_loc_exp_batch.py \
        --exp-dir /data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/multi_turn_loc_exp_vanilla \
        --dataset $DATASET \
        --dataset-path $DATASET_PATH \
        --device cuda:0 \
        
    if [ $? -ne 0 ]; then
        FAILED_EXPERIMENTS+=("multi_turn_loc_exp_vanilla_$DATASET")
    fi
done

CUDA_VISIBLE_DEVICES=5 python eval_multi_turn_loc_exp_batch.py --exp-dir /data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/multi_turn_loc_exp_vanilla --dataset val --dataset-path /home/jhpark/image-artifacts/src/train/LLaMA-Factory/data/multi_turn_loc_exp_val.json  
if [ $? -ne 0 ]; then
    FAILED_EXPERIMENTS+=("multi_turn_loc_exp_vanilla_val")
fi

for DATASET in "${DATASETS[@]}"
do
    echo "Evaluating dataset: $DATASET"
    
    # Set dataset-specific path
    case $DATASET in
        "synthscars")
            DATASET_PATH="/data2/jhpark/image-artifacts/data/eval/SynthScars/test"
            ;;
        "synartifact")
            DATASET_PATH="/data2/jhpark/image-artifacts/data/eval/SynArtifact"
            ;;
        "loki")
            DATASET_PATH="/data2/jhpark/image-artifacts/data/eval/loki"
            ;;
        "richhf")
            DATASET_PATH="/data2/jhpark/image-artifacts/data/eval/richhf-18k"
            ;;
        *)
            DATASET_PATH="/data2/jhpark/image-artifacts/data/eval"
            ;;
    esac
    
    CUDA_VISIBLE_DEVICES=5 python eval_multi_turn_loc_exp_batch.py \
        --exp-dir /data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/1000/multi_turn_loc_exp_fireflow \
        --dataset $DATASET \
        --dataset-path $DATASET_PATH \
        --device cuda:0 \
        
    if [ $? -ne 0 ]; then
        FAILED_EXPERIMENTS+=("multi_turn_loc_exp_fireflow_$DATASET")
    fi
done

CUDA_VISIBLE_DEVICES=5 python eval_multi_turn_loc_exp_batch.py --exp-dir /data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/1000/multi_turn_loc_exp_fireflow --dataset val --dataset-path /home/jhpark/image-artifacts/src/train/LLaMA-Factory/data/multi_turn_loc_exp_val.json  
if [ $? -ne 0 ]; then
    FAILED_EXPERIMENTS+=("multi_turn_loc_exp_fireflow_val")
fi

# Print summary of failed experiments
echo ""
echo "=========================================="
echo "EXPERIMENT SUMMARY"
echo "=========================================="

if [ ${#FAILED_EXPERIMENTS[@]} -eq 0 ]; then
    echo "✅ All experiments completed successfully!"
else
    echo "❌ The following experiments failed:"
    for experiment in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  - $experiment"
    done
    echo ""
    echo "Total failed experiments: ${#FAILED_EXPERIMENTS[@]}"
fi

echo "=========================================="