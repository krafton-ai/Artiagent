"""
Comprehensive Evaluation Script

This script runs all three evaluation types (binary, localization, explanation)
at once and generates a comprehensive report with all metrics.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def setup_logging(output_dir: str, exp_name: str) -> str:
    """Setup logging for comprehensive evaluation."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"comprehensive_eval_{exp_name}_{timestamp}.log"
    log_file = output_path / log_filename
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return str(log_file)


def run_evaluation(script_path: str, args: Dict) -> Dict:
    """
    Run a single evaluation script and return results.
    
    Args:
        script_path: Path to evaluation script
        args: Dictionary of arguments
        
    Returns:
        Dictionary with evaluation results
    """
    # Build command
    cmd = [sys.executable, script_path]
    for key, value in args.items():
        if value is not None:
            cmd.append(f"--{key}")
            if not isinstance(value, bool):
                cmd.append(str(value))
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        # Run evaluation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"Evaluation completed successfully")
        logger.debug(result.stdout)
        
        # Parse output to find results file
        lines = result.stdout.split('\n')
        results_file = None
        for line in lines:
            if 'Results saved to:' in line or 'results saved to:' in line.lower():
                # Extract file path
                parts = line.split(':')
                if len(parts) >= 2:
                    results_file = parts[-1].strip()
                    break
        
        if results_file and Path(results_file).exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Could not find results file in output")
            return {'error': 'Results file not found', 'stdout': result.stdout}
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed with error: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return {'error': str(e), 'stderr': e.stderr}


def run_comprehensive_evaluation(
    exp_dir: str,
    dataset: str,
    format_type: str,
    dataset_path: str,
    device: str,
    batch_size: int,
    max_samples: int,
    output_dir: str
):
    """
    Run all three evaluation types and generate comprehensive report.
    
    Args:
        exp_dir: Path to model checkpoint
        dataset: Dataset name
        format_type: Model format (original, single_vqa, multi_vqa)
        dataset_path: Path to evaluation dataset
        device: CUDA device
        batch_size: Batch size (ignored for multi_vqa)
        max_samples: Maximum samples to evaluate
        output_dir: Output directory for results
    """
    
    exp_name = Path(exp_dir).name
    log_file = setup_logging(output_dir, exp_name)
    
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Model: {exp_dir}")
    logger.info(f"Format: {format_type}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Max samples: {max_samples if max_samples else 'All'}")
    logger.info("=" * 80)
    
    # Determine which evaluation script to use
    script_dir = Path(__file__).parent
    if format_type == "original":
        eval_script = script_dir / "eval_finetuned_batch.py"
    elif format_type == "single_vqa":
        eval_script = script_dir / "eval_single_vqa_batch.py"
    elif format_type == "multi_vqa":
        eval_script = script_dir / "eval_multi_vqa_batch.py"
    else:
        raise ValueError(f"Unknown format type: {format_type}")
    
    if not eval_script.exists():
        raise FileNotFoundError(f"Evaluation script not found: {eval_script}")
    
    # Evaluation types to run
    eval_types = ["binary", "localization", "explanation"]
    
    all_results = {}
    all_metrics = {}
    
    # Run each evaluation type
    for eval_type in eval_types:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Running {eval_type.upper()} evaluation...")
        logger.info("=" * 80)
        
        # Prepare arguments
        eval_args = {
            "exp-dir": exp_dir,
            "dataset": dataset,
            "type": eval_type,
            "dataset-path": dataset_path,
            "device": device,
            "max-samples": max_samples,
        }
        
        # Add batch-size only for formats that support it
        if format_type != "multi_vqa":
            eval_args["batch-size"] = batch_size
        
        # Run evaluation
        results = run_evaluation(str(eval_script), eval_args)
        
        if 'error' in results:
            logger.error(f"Failed to run {eval_type} evaluation: {results['error']}")
            all_results[eval_type] = results
            continue
        
        # Extract metrics
        metrics = results.get('metrics', {})
        all_results[eval_type] = results
        all_metrics[eval_type] = metrics
        
        # Log metrics
        logger.info(f"\n{eval_type.upper()} Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    # Generate comprehensive summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Model: {exp_name}")
    logger.info(f"Format: {format_type}")
    logger.info(f"Dataset: {dataset}")
    logger.info("=" * 80)
    
    # Binary metrics
    if 'binary' in all_metrics:
        logger.info("\nBINARY CLASSIFICATION:")
        binary_acc = all_metrics['binary'].get('binary_accuracy', 0.0)
        logger.info(f"  Accuracy: {binary_acc:.4f} ({binary_acc*100:.2f}%)")
        logger.info(f"  Samples: {all_metrics['binary'].get('total_samples', 0)}")
    
    # Localization metrics
    if 'localization' in all_metrics:
        logger.info("\nLOCALIZATION:")
        mean_iou = all_metrics['localization'].get('mean_iou', 0.0)
        mean_f1 = all_metrics['localization'].get('mean_f1', 0.0)
        mean_precision = all_metrics['localization'].get('mean_precision', 0.0)
        mean_recall = all_metrics['localization'].get('mean_recall', 0.0)
        
        logger.info(f"  Mean IoU: {mean_iou:.4f}")
        logger.info(f"  Mean F1: {mean_f1:.4f}")
        logger.info(f"  Mean Precision: {mean_precision:.4f}")
        logger.info(f"  Mean Recall: {mean_recall:.4f}")
        logger.info(f"  Valid samples: {all_metrics['localization'].get('valid_samples', 0)}")
        
        # Additional metrics if available
        if 'mean_legion_iou' in all_metrics['localization']:
            logger.info(f"  LEGION IoU: {all_metrics['localization']['mean_legion_iou']:.4f}")
        if 'mean_wsol_iou' in all_metrics['localization']:
            logger.info(f"  WSOL IoU: {all_metrics['localization']['mean_wsol_iou']:.4f}")
    
    # Explanation metrics
    if 'explanation' in all_metrics:
        logger.info("\nEXPLANATION:")
        mean_rouge = all_metrics['explanation'].get('mean_rouge_l', 0.0)
        mean_css = all_metrics['explanation'].get('mean_css', 0.0)
        
        logger.info(f"  Mean ROUGE-L: {mean_rouge:.4f}")
        logger.info(f"  Mean CSS: {mean_css:.4f}")
        logger.info(f"  Samples: {all_metrics['explanation'].get('total_samples', 0)}")
        
        # Valid samples if available
        if 'valid_samples_count' in all_metrics['explanation']:
            valid_rouge = all_metrics['explanation'].get('valid_mean_rouge_l', 0.0)
            valid_css = all_metrics['explanation'].get('valid_mean_css', 0.0)
            logger.info(f"\n  Valid (positive) samples only:")
            logger.info(f"    Count: {all_metrics['explanation']['valid_samples_count']}")
            logger.info(f"    Mean ROUGE-L: {valid_rouge:.4f}")
            logger.info(f"    Mean CSS: {valid_css:.4f}")
    
    logger.info("=" * 80)
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(output_dir) / f"comprehensive_{format_type}_{dataset}_{exp_name}_{timestamp}.json"
    
    comprehensive_results = {
        'config': {
            'exp_dir': exp_dir,
            'format_type': format_type,
            'dataset': dataset,
            'max_samples': max_samples,
            'batch_size': batch_size if format_type != "multi_vqa" else None,
            'timestamp': timestamp
        },
        'summary_metrics': all_metrics,
        'detailed_results': all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    logger.info(f"\n✅ Comprehensive results saved to: {results_file}")
    
    # Create a simple CSV summary for easy comparison
    csv_file = Path(output_dir) / f"summary_{format_type}_{dataset}_{exp_name}_{timestamp}.csv"
    with open(csv_file, 'w') as f:
        f.write("Metric,Value\n")
        
        # Binary
        if 'binary' in all_metrics:
            f.write(f"binary_accuracy,{all_metrics['binary'].get('binary_accuracy', 0.0):.4f}\n")
        
        # Localization
        if 'localization' in all_metrics:
            f.write(f"localization_iou,{all_metrics['localization'].get('mean_iou', 0.0):.4f}\n")
            f.write(f"localization_f1,{all_metrics['localization'].get('mean_f1', 0.0):.4f}\n")
            f.write(f"localization_precision,{all_metrics['localization'].get('mean_precision', 0.0):.4f}\n")
            f.write(f"localization_recall,{all_metrics['localization'].get('mean_recall', 0.0):.4f}\n")
        
        # Explanation
        if 'explanation' in all_metrics:
            f.write(f"explanation_rouge_l,{all_metrics['explanation'].get('mean_rouge_l', 0.0):.4f}\n")
            f.write(f"explanation_css,{all_metrics['explanation'].get('mean_css', 0.0):.4f}\n")
    
    logger.info(f"✅ CSV summary saved to: {csv_file}")
    logger.info(f"✅ Log file saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation (all metrics) for a model"
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to experiment directory (model checkpoint)"
    )
    parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=["original", "single_vqa", "multi_vqa"],
        help="Model format type"
    )
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
        help="Batch size (ignored for multi_vqa format)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None = all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: ./eval_results)"
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).parent / "eval_results")
    
    # Run comprehensive evaluation
    run_comprehensive_evaluation(
        exp_dir=args.exp_dir,
        dataset=args.dataset,
        format_type=args.format,
        dataset_path=args.dataset_path,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

