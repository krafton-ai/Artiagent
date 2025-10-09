#!/bin/bash

# =============================================================================
# Quick LEGION Test: Generate + Evaluate (Limited Samples)
# =============================================================================
# A simplified version for quick testing with limited samples
# 
# Usage:
#   ./quick_legion_test.sh [dataset] [max_samples]
#
# Examples:
#   ./quick_legion_test.sh                    # synthscars, 2 samples  
#   ./quick_legion_test.sh loki 3             # loki dataset, 3 samples
# =============================================================================

set -e  # Exit on any error

# Configuration
DATASET="${1:-synthscars}"
MAX_SAMPLES="${2:-2}"
GENERATION_ENV="legion1.4.7"
EVALUATION_ENV="lfac"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_step() {
    echo -e "\n${BLUE}🚀 $1${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Main execution
print_step "Quick LEGION Test Pipeline"
print_info "Dataset: ${DATASET}"
print_info "Max samples: ${MAX_SAMPLES}"

# Phase 1: Generation
print_step "PHASE 1: Generating responses in ${GENERATION_ENV} environment"
conda activate "${GENERATION_ENV}"
./generate_all_legion_responses.sh
print_success "Generation completed!"

# Phase 2: Evaluation  
print_step "PHASE 2: Running evaluation in ${EVALUATION_ENV} environment"
conda activate "${EVALUATION_ENV}"

# Test localization evaluation
python eval_with_pregenerated.py --model legion --dataset "${DATASET}" --type localization --max_samples "${MAX_SAMPLES}"
print_success "Localization evaluation completed!"

# Test explanation evaluation  
python eval_with_pregenerated.py --model legion --dataset "${DATASET}" --type explanation --max_samples "${MAX_SAMPLES}"
print_success "Explanation evaluation completed!"

print_step "✅ Quick test completed successfully!"
print_info "Check eval_logs/logs/ for detailed results"
