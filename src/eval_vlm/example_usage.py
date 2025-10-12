"""
Example usage of the finetuned VLM evaluation system.

This script demonstrates how to use the evaluation tools with your Qwen 2.5 VL checkpoint.
"""

import subprocess
import sys
from pathlib import Path


def run_example_evaluation():
    """Run example evaluation with the Qwen 2.5 VL checkpoint."""
    
    # Your checkpoint path
    checkpoint_path = "/data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/full/sft_vanilla_random_2k/checkpoint-1000"
    
    # Dataset base directory
    base_dir = "/data2/jhpark/image-artifacts/data/eval"
    
    print("🚀 Running example evaluation with Qwen 2.5 VL finetuned model")
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"📊 Dataset base: {base_dir}")
    
    # Example 1: Test with a few samples
    print("\n" + "="*60)
    print("Example 1: Test evaluation with 5 samples")
    print("="*60)
    
    cmd = [
        sys.executable, "test_evaluation.py",
        "--exp-dir", checkpoint_path,
        "--max-samples", "5",
        "--dataset", "ours",
        "--eval-type", "explanation"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Test evaluation completed successfully!")
        else:
            print(f"❌ Test evaluation failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error running test evaluation: {e}")
    
    # Example 2: Run evaluation on explanation task with batch processing
    print("\n" + "="*60)
    print("Example 2: Batch evaluation on explanation task")
    print("="*60)
    
    cmd = [
        sys.executable, "eval_finetuned.py",
        "--exp-dir", checkpoint_path,
        "--dataset", "ours",
        "--type", "explanation",
        "--base-dir", base_dir,
        "--device", "cuda:4",
        "--max-samples", "10",
        "--use-batch",
        "--batch-size", "4"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Explanation evaluation completed successfully!")
        else:
            print(f"❌ Explanation evaluation failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error running explanation evaluation: {e}")
    
    # Example 3: Run comprehensive evaluation with batch processing
    print("\n" + "="*60)
    print("Example 3: Comprehensive batch evaluation (multiple datasets and types)")
    print("="*60)
    
    cmd = [
        sys.executable, "run_evaluation.py",
        "--exp-dir", checkpoint_path,
        "--comprehensive",
        "--datasets", "ours", "synthscars",
        "--eval-types", "binary", "explanation",
        "--base-dir", base_dir,
        "--device", "cuda:4",
        "--max-samples", "5",
        "--use-batch",
        "--batch-size", "2"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Comprehensive evaluation completed successfully!")
        else:
            print(f"❌ Comprehensive evaluation failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error running comprehensive evaluation: {e}")


if __name__ == "__main__":
    run_example_evaluation()
