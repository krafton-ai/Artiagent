#!/bin/bash

# Legion evaluation using pre-generated responses
# This script should be run in the lfac environment from the src/eval/legion/ directory

echo "🚀 Starting LEGION evaluations using pre-generated responses..."

# Check if pre-generated responses exist
RESPONSES_DIR="/data2/jhpark/image-artifacts/eval/refined_legion_responses"
if [ ! -d "$RESPONSES_DIR" ]; then
    echo "❌ Pre-generated responses directory not found: $RESPONSES_DIR"
    echo "Please run generate_legion_responses.py in the legion1.4.7 environment first."
    exit 1
fi

# Check for response files
MISSING_FILES=()
for dataset in "synthscars" "synartifact" "loki" "richhf" "ours"; do
    if [ ! -f "$RESPONSES_DIR/${dataset}_responses.pkl" ]; then
        MISSING_FILES+=("${dataset}_responses.pkl")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "⚠️  Some pre-generated response files are missing:"
    printf '%s\n' "${MISSING_FILES[@]}"
    echo "Continuing with available datasets..."
fi

echo ""
echo "=== WSOL Evaluation (Localization 1) ==="
if [ -f "$RESPONSES_DIR/synthscars_responses.pkl" ]; then
    echo "📊 Running WSOL eval on SynthScars..."
    python eval_with_pregenerated.py --model 'legion' --dataset 'synthscars' --type 'localization'
fi

if [ -f "$RESPONSES_DIR/synartifact_responses.pkl" ]; then
    echo "📊 Running WSOL eval on SynArtifact..."
    python eval_with_pregenerated.py --model 'legion' --dataset 'synartifact' --type 'localization'
fi

if [ -f "$RESPONSES_DIR/loki_responses.pkl" ]; then
    echo "📊 Running WSOL eval on LOKI..."
    python eval_with_pregenerated.py --model 'legion' --dataset 'loki' --type 'localization'
fi

if [ -f "$RESPONSES_DIR/richhf_responses.pkl" ]; then
    echo "📊 Running WSOL eval on RichHF..."
    python eval_with_pregenerated.py --model 'legion' --dataset 'richhf' --type 'localization'
fi

if [ -f "$RESPONSES_DIR/ours_responses.pkl" ]; then
    echo "📊 Running WSOL eval on OURS..."
    python eval_with_pregenerated.py --model 'legion' --dataset 'ours' --type 'localization'
fi

echo ""
echo "=== Explanation Evaluation ==="
if [ -f "$RESPONSES_DIR/synthscars_responses.pkl" ]; then
    echo "📊 Running explanation eval on SynthScars..."
    python eval_with_pregenerated.py --model 'legion' --dataset 'synthscars' --type 'explanation'
fi

if [ -f "$RESPONSES_DIR/loki_responses.pkl" ]; then
    echo "📊 Running explanation eval on LOKI..."
    python eval_with_pregenerated.py --model 'legion' --dataset 'loki' --type 'explanation'
fi

if [ -f "$RESPONSES_DIR/ours_responses.pkl" ]; then
    echo "📊 Running explanation eval on OURS..."
    python eval_with_pregenerated.py --model 'legion' --dataset 'ours' --type 'explanation'
fi

echo ""
echo "🎉 All LEGION evaluations completed!"
echo "📊 Check the output logs above for detailed results."
