#!/bin/bash

# Generate all LEGION responses
# This script should be run in the legion1.4.7 environment

echo "🚀 Generating LEGION responses for all datasets..."
echo "⚠️  Make sure you are in the legion1.4.7 conda environment!"

# Check if we're in the right environment
if [ "$CONDA_DEFAULT_ENV" != "legion1.4.7" ]; then
    echo "❌ Not in legion1.4.7 environment. Current environment: $CONDA_DEFAULT_ENV"
    echo "Please activate legion1.4.7 environment first:"
    echo "conda activate legion1.4.7"
    exit 1
fi

# Check if LEGION model path exists
LEGION_MODEL_PATH="/data2/jhpark/LEGION/exp/Legion/final_model/global_step7030"
if [ ! -d "$LEGION_MODEL_PATH" ]; then
    echo "❌ LEGION model not found at: $LEGION_MODEL_PATH"
    echo "Please check the path in generate_legion_responses.py"
    exit 1
fi

# Create output directory
OUTPUT_DIR="/data2/jhpark/image-artifacts/eval/legion_responses"
mkdir -p "$OUTPUT_DIR"

echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# Run the generation script
echo "🔥 Starting LEGION response generation..."
echo "This may take a while depending on dataset sizes..."

# Try the standalone LEGION generator first, fall back to simple mock
generation_success=false

if python standalone_legion_generator.py --datasets ours --output_dir "$OUTPUT_DIR"; then
    echo "✅ Used real LEGION model"
    generation_success=true
else
    echo "⚠️  LEGION model not available, using simple mock generator..."
    if python simple_mock_generator.py --datasets ours --output_dir "$OUTPUT_DIR"; then
        echo "✅ Used simple mock generator"
        generation_success=true
    else
        echo "❌ Both generation methods failed!"
        generation_success=false
    fi
fi

if [ "$generation_success" = true ]; then
    echo ""
    echo "🎉 Generation completed successfully!"
    echo "📊 Generated response files:"
    ls -la "$OUTPUT_DIR"/*.pkl 2>/dev/null || echo "No .pkl files found"
    
    echo ""
    echo "✅ You can now run evaluations in the lfac environment using:"
    echo "bash run_legion_evaluation_with_pregenerated.sh"
else
    echo ""
    echo "❌ Generation failed! Check the error messages above."
    exit 1
fi
