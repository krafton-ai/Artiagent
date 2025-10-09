#!/bin/bash

# =============================================================================
# Simple LEGION Pipeline: Generate + Evaluate
# =============================================================================
# This script simply runs the two main scripts in sequence:
# 1. generate_all_legion_responses.sh (in legion1.4.7 environment)
# 2. run_legion_evaluation_with_pregenerated.sh (in lfac environment)
# =============================================================================

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_step() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# =============================================================================
# Main Pipeline
# =============================================================================

print_step "LEGION COMPLETE PIPELINE"

# Phase 1: Generate LEGION responses
print_step "PHASE 1: GENERATING LEGION RESPONSES"
print_info "Switching to legion1.4.7 environment for LEGION model..."

# Activate legion1.4.7 environment for generation
source $(conda info --base)/etc/profile.d/conda.sh
conda activate legion1.4.7

if ./generate_all_legion_responses.sh; then
    print_success "LEGION response generation completed successfully!"
else
    print_error "LEGION response generation failed!"
    exit 1
fi

# Phase 2: Run evaluations with pre-generated responses
print_step "PHASE 2: RUNNING EVALUATIONS WITH PRE-GENERATED RESPONSES"
print_info "Switching to lfac environment for evaluations..."

# Activate lfac environment for evaluation
conda activate lfac

if ./run_legion_evaluation_with_pregenerated.sh; then
    print_success "LEGION evaluations completed successfully!"
else
    print_error "LEGION evaluations failed!"
    exit 1
fi

# Show log summary
print_step "PIPELINE COMPLETED SUCCESSFULLY!"
print_success "Both generation and evaluation phases completed."

print_info "Recent log files:"
if [ -d "eval_logs/logs" ]; then
    ls -la eval_logs/logs/*.log | tail -5
else
    print_info "No log directory found"
fi

print_info "\nTo view latest results, check:"
print_info "  tail -50 eval_logs/logs/<latest_log_file>"
