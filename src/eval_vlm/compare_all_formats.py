"""
Compare All Formats Script

This script evaluates all three model formats (original, single_vqa, multi_vqa)
on the same dataset and generates a comparison report.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd


def run_comprehensive_eval(exp_dir: str, format_type: str, dataset: str, 
                          dataset_path: str, device: str, batch_size: int,
                          max_samples: int) -> Dict:
    """Run comprehensive evaluation for a single model."""
    
    script_path = Path(__file__).parent / "eval_comprehensive.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--exp-dir", exp_dir,
        "--format", format_type,
        "--dataset", dataset,
        "--dataset-path", dataset_path,
        "--device", device,
        "--batch-size", str(batch_size),
    ]
    
    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])
    
    print(f"\n{'='*80}")
    print(f"Evaluating {format_type.upper()} format: {Path(exp_dir).name}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        
        # Parse output to find results file
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Comprehensive results saved to:' in line:
                results_file = line.split(':')[-1].strip()
                if Path(results_file).exists():
                    with open(results_file, 'r') as f:
                        return json.load(f)
        
        return {'error': 'Could not find results file'}
    
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        print(f"STDERR: {e.stderr}")
        return {'error': str(e)}


def compare_formats(
    original_dir: str,
    single_vqa_dir: str,
    multi_vqa_dir: str,
    dataset: str,
    dataset_path: str,
    device: str,
    batch_size: int,
    max_samples: int,
    output_dir: str
):
    """
    Compare all three formats.
    
    Args:
        original_dir: Path to original format model
        single_vqa_dir: Path to single-turn VQA model
        multi_vqa_dir: Path to multi-turn VQA model
        dataset: Dataset name
        dataset_path: Path to evaluation dataset
        device: CUDA device
        batch_size: Batch size
        max_samples: Maximum samples to evaluate
        output_dir: Output directory
    """
    
    print("="*80)
    print("COMPARING ALL VQA FORMATS")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Max samples: {max_samples if max_samples else 'All'}")
    print("="*80)
    
    results = {}
    
    # Evaluate each format
    formats = [
        ("original", original_dir),
        ("single_vqa", single_vqa_dir),
        ("multi_vqa", multi_vqa_dir)
    ]
    
    for format_name, exp_dir in formats:
        if exp_dir:
            result = run_comprehensive_eval(
                exp_dir, format_name, dataset, dataset_path,
                device, batch_size, max_samples
            )
            results[format_name] = result
        else:
            print(f"⚠️  Skipping {format_name} (no directory provided)")
            results[format_name] = None
    
    # Generate comparison report
    print("\n" + "="*80)
    print("COMPARISON REPORT")
    print("="*80)
    
    # Prepare comparison data
    comparison_data = []
    
    for format_name, result in results.items():
        if result is None or 'error' in result:
            continue
        
        metrics = result.get('summary_metrics', {})
        
        row = {
            'Format': format_name.replace('_', ' ').title(),
            'Model': Path(result['config']['exp_dir']).name if result.get('config') else 'N/A'
        }
        
        # Binary metrics
        if 'binary' in metrics:
            row['Binary Accuracy'] = metrics['binary'].get('binary_accuracy', 0.0)
        
        # Localization metrics
        if 'localization' in metrics:
            row['Loc IoU'] = metrics['localization'].get('mean_iou', 0.0)
            row['Loc F1'] = metrics['localization'].get('mean_f1', 0.0)
            row['Loc Precision'] = metrics['localization'].get('mean_precision', 0.0)
            row['Loc Recall'] = metrics['localization'].get('mean_recall', 0.0)
        
        # Explanation metrics
        if 'explanation' in metrics:
            row['Expl ROUGE-L'] = metrics['explanation'].get('mean_rouge_l', 0.0)
            row['Expl CSS'] = metrics['explanation'].get('mean_css', 0.0)
        
        comparison_data.append(row)
    
    if comparison_data:
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Print table
        print("\n" + df.to_string(index=False, float_format='%.4f'))
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = Path(output_dir) / f"format_comparison_{dataset}_{timestamp}.csv"
        df.to_csv(csv_file, index=False, float_format='%.4f')
        print(f"\n✅ Comparison CSV saved to: {csv_file}")
        
        # Save detailed JSON
        json_file = Path(output_dir) / f"format_comparison_{dataset}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'dataset': dataset,
                'comparison': comparison_data,
                'detailed_results': results
            }, f, indent=2)
        print(f"✅ Detailed JSON saved to: {json_file}")
        
        # Print summary insights
        print("\n" + "="*80)
        print("SUMMARY INSIGHTS")
        print("="*80)
        
        if 'Binary Accuracy' in df.columns:
            best_binary = df.loc[df['Binary Accuracy'].idxmax()]
            print(f"\n✨ Best Binary Classification: {best_binary['Format']} ({best_binary['Binary Accuracy']:.4f})")
        
        if 'Loc IoU' in df.columns:
            best_loc = df.loc[df['Loc IoU'].idxmax()]
            print(f"✨ Best Localization (IoU): {best_loc['Format']} ({best_loc['Loc IoU']:.4f})")
        
        if 'Loc F1' in df.columns:
            best_f1 = df.loc[df['Loc F1'].idxmax()]
            print(f"✨ Best Localization (F1): {best_f1['Format']} ({best_f1['Loc F1']:.4f})")
        
        if 'Expl ROUGE-L' in df.columns:
            best_rouge = df.loc[df['Expl ROUGE-L'].idxmax()]
            print(f"✨ Best Explanation (ROUGE-L): {best_rouge['Format']} ({best_rouge['Expl ROUGE-L']:.4f})")
        
        if 'Expl CSS' in df.columns:
            best_css = df.loc[df['Expl CSS'].idxmax()]
            print(f"✨ Best Explanation (CSS): {best_css['Format']} ({best_css['Expl CSS']:.4f})")
        
        print("="*80)
    else:
        print("⚠️  No valid results to compare")


def main():
    parser = argparse.ArgumentParser(
        description="Compare all three VQA format models on the same dataset"
    )
    
    # Model directories
    parser.add_argument(
        "--original-dir",
        type=str,
        default=None,
        help="Path to original format model checkpoint"
    )
    parser.add_argument(
        "--single-vqa-dir",
        type=str,
        default=None,
        help="Path to single-turn VQA model checkpoint"
    )
    parser.add_argument(
        "--multi-vqa-dir",
        type=str,
        default=None,
        help="Path to multi-turn VQA model checkpoint"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="ours",
        choices=["ours", "t2i", "synartifact"],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/data2/jhpark/image-artifacts/data/eval",
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate per format"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ./eval_results)"
    )
    
    args = parser.parse_args()
    
    # Check that at least one model directory is provided
    if not any([args.original_dir, args.single_vqa_dir, args.multi_vqa_dir]):
        parser.error("At least one model directory must be provided")
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).parent / "eval_results")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run comparison
    compare_formats(
        original_dir=args.original_dir,
        single_vqa_dir=args.single_vqa_dir,
        multi_vqa_dir=args.multi_vqa_dir,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

