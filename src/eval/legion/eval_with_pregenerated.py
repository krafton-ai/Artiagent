"""
Patched evaluation functions that support pre-generated LEGION responses.

This module provides modified versions of the evaluation functions that:
1. Set the current image path context for mock LEGION models
2. Work with pre-generated responses seamlessly
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the models with context functions
from models import set_current_image_path, get_current_image_path

def setup_logging(output_dir: str, dataset_type: str, model: str, use_finetuned: bool, eval_type: str) -> logging.Logger:
    """
    Setup logging configuration with file and console handlers.
    
    Args:
        output_dir: Directory where logs will be saved
        dataset_type: Dataset name for log file naming
        model: Model name for log file naming
        use_finetuned: Whether using finetuned model
        eval_type: Type of evaluation for log file naming
        
    Returns:
        Configured logger instance
    """
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if use_finetuned:
        if eval_type == 'localization':
            log_file = log_dir / f'pregenerated_eval_{dataset_type}_{model}_finetuned_bbox.log'
        else:
            log_file = log_dir / f'pregenerated_eval_{dataset_type}_{model}_finetuned_{eval_type}.log'
    else:
        if eval_type == 'localization':
            log_file = log_dir / f'pregenerated_eval_{dataset_type}_{model}_bbox_{timestamp}.log'
        else:
            log_file = log_dir / f'pregenerated_eval_{dataset_type}_{model}_{eval_type}_{timestamp}.log'
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def patched_unified_inference(model, image: Image.Image, prompt: str, image_path: str = None) -> Dict[str, Any]:
    """
    Patched unified inference that sets image path context for mock models.
    
    Args:
        model: Model instance 
        image: PIL Image to analyze
        prompt: Text prompt for inference
        image_path: Path to image file (for mock LEGION)
        
    Returns:
        Dictionary containing standardized inference results
    """
    # Import here to avoid circular imports
    from models import PalEval, DiffEval, LegionEval
    from eval_utils import parse_json
    
    # Set image path context if provided
    if image_path is not None:
        set_current_image_path(image_path)
    
    try:
        # Handle models that only take image (no prompt)
        if isinstance(model, (PalEval, DiffEval, LegionEval)):
            result = model.inference(image)
        else:
            # Models that take both image and prompt
            result = model.inference(image, prompt)
        
        # Handle None returns (GPTEval can return None on error)
        if result is None:
            return {"error": "model_returned_none", "raw_response": ""}
        
        # Handle string returns (QwenEval returns string)
        if isinstance(result, str):
            try:
                # Try to parse the string as JSON
                parsed_result = parse_json(result)
                return {"parsed_output": parsed_result, "raw_response": result}
            except Exception as e:
                # If parsing fails, return as raw response
                return {"error": "json_parse_failed", "raw_response": result, "parse_error": str(e)}
        
        # Handle dictionary returns (most models)
        if isinstance(result, dict):
            # Check if it already has the expected structure
            if "raw_response" in result or "error" in result or "heatmap" in result:
                return result
            else:
                # Wrap in standard format
                return {"parsed_output": result, "raw_response": str(result)}
        
        # Handle other types
        return {"error": "unexpected_return_type", "raw_response": str(result), "type": type(result).__name__}
        
    except Exception as e:
        return {"error": "inference_exception", "raw_response": "", "exception": str(e)}


def patched_run_evaluation(config: Dict, max_samples: Optional[int] = None, script_type: str = "eval"):
    """
    Patched run evaluation that works with pre-generated LEGION responses.
    
    Args:
        config: Configuration dictionary
        max_samples: Maximum number of samples to evaluate (None for all)
        script_type: Type of evaluation script ("eval", "legion_eval", "wsol_eval")
    """
    # Import modules based on script type
    if script_type == "eval":
        from eval import DatasetIterator, create_model, extract_prediction_result, Evaluation
        import eval_utils
        import legion_eval_utils  
        import wsol_eval_utils
        create_prompt = eval_utils.create_prompt
        legion_evaluator = legion_eval_utils.Evaluation()
        wsol_evaluator = wsol_eval_utils.Evaluation()
    elif script_type == "legion_eval":
        from legion_eval import DatasetIterator, create_model, extract_prediction_result, Evaluation
        from legion_eval_utils import create_prompt
        legion_evaluator = wsol_evaluator = None
    elif script_type == "wsol_eval":
        from wsol_eval import DatasetIterator, create_model, extract_prediction_result
        from wsol_eval_utils import Evaluation, create_prompt
        from eval_with_visualization import VisualizationIntegratedEvaluator
        legion_evaluator = wsol_evaluator = None
    else:
        raise ValueError(f"Unknown script type: {script_type}")
    
    # Import model classes for isinstance checks
    from models import PalEval, GPTEval
    
    logger = logging.getLogger(__name__)
    dataset_type = config['dataset_type']
    eval_type = config['eval_type']
    
    logger.info(f"Starting evaluation for {dataset_type} dataset")
    if config['use_finetuned']:
        logger.info("Using finetuned model")
    else:
        logger.info("Running zero-shot evaluation")
    
    # Initialize components
    logger.info("Initializing model and data iterator...")
    model = create_model(config)
    data_iterator = DatasetIterator(config)
    
    if script_type == "eval":
        evaluator = Evaluation()
    elif script_type == "legion_eval":
        evaluator = Evaluation()
    elif script_type == "wsol_eval":
        evaluator = Evaluation()
    
    if script_type == "eval":
        # Create additional evaluators for comprehensive localization evaluation
        legion_evaluator = legion_eval_utils.Evaluation()
        wsol_evaluator = wsol_eval_utils.Evaluation()

    if config.get('model_type') == 'pal' and isinstance(model, PalEval):
        memory_info = model.get_gpu_memory_info()
        for device, info in memory_info.items():
            logger.info(f"📊 GPU {device} - Allocated: {info['allocated_gb']:.2f}GB, Reserved: {info['cached_gb']:.2f}GB")
    
    # Determine number of samples to process
    total_samples = len(data_iterator)
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)
    
    # Get available pre-generated responses to filter dataset (only for pregenerated version)
    if hasattr(model, '_mock') and hasattr(model._mock, 'get_available_image_keys'):
        available_keys = set(model._mock.get_available_image_keys())
        logger.info(f"📊 Found {len(available_keys)} pre-generated responses")
        logger.info(f"📋 Available responses: {list(available_keys)[:5]}{'...' if len(available_keys) > 5 else ''}")
        actual_samples = min(len(available_keys), total_samples) if max_samples is not None else len(available_keys)
        logger.info(f"Processing {actual_samples} samples with pre-generated responses")
    else:
        available_keys = set()
        logger.info(f"Processing {total_samples} samples")

    results = {}

    prompt = create_prompt(eval_type)
    logger.info(f"Input query: {prompt}")
    
    # Process samples
    for i, (json_data, image_path) in enumerate(data_iterator):
        if max_samples is not None and i >= max_samples:
            break
            
        try:
            # Check if this image has a pre-generated response (only for pregenerated version)
            if available_keys:
                image_key = image_path.name if hasattr(image_path, 'name') else Path(str(image_path)).name
                if image_key not in available_keys:
                    continue
            
            logger.info(f"Processing sample {i+1}/{total_samples}: {image_path}")

            # Load and process image
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue

            image = Image.open(str(image_path)).convert("RGB")
            if dataset_type == 'richhf':
                image = image.resize((512, 512), Image.LANCZOS)

            # Run model inference with unified interface (patched for pregenerated responses)
            unified_output = patched_unified_inference(model, image, prompt, str(image_path))
            print(f"Unified output: {unified_output}")
            
            prediction = extract_prediction_result(unified_output, config['use_finetuned'], eval_type)
            print(f"Extracted prediction: {prediction}")
            
            # Evaluate results  
            if script_type == "eval":
                stats = evaluator.generate_statistics(
                    dataset_type, eval_type, json_data, prediction, image_size=image.size
                )
                
                # For localization evaluation, also run LEGION and WSOL methods
                if eval_type == 'localization':
                    legion_stats = legion_evaluator.generate_statistics(
                        dataset_type, eval_type, json_data, prediction, image_size=image.size
                    )
                    wsol_stats = wsol_evaluator.generate_statistics(
                        dataset_type, eval_type, json_data, prediction, image_size=image.size
                    )
                    
                    # Merge stats with prefixes to distinguish evaluation methods
                    stats.update({f'legion_{k}': v for k, v in legion_stats.items() if k not in ['binary_success', 'rouge_l', 'css', 'classification', 'has_gt_artifacts', 'has_pred_artifacts']})
                    stats.update({f'wsol_{k}': v for k, v in wsol_stats.items() if k not in ['binary_success', 'rouge_l', 'css', 'classification', 'has_gt_artifacts', 'has_pred_artifacts']})
                    
            else:
                # For legion_eval and wsol_eval
                stats = evaluator.generate_statistics(
                    dataset_type, eval_type, json_data, prediction, image_size=image.size
                )
                
            if eval_type == 'binary':
                sample_result = {
                    'image_path': str(image_path),
                    'binary_success': stats['binary_success'],
                    'classification': stats['classification'],
                    'has_gt_artifacts': stats['has_gt_artifacts'],
                    'has_pred_artifacts': stats['has_pred_artifacts'],
                    'prediction': prediction
                }
                logger.info(
                    f"Sample {i + 1} - Binary: {sample_result['binary_success']}, "
                    f"Prediction: {prediction}"
                )
            elif eval_type == 'localization':
                sample_result = {
                    'image_path': str(image_path),
                    # Standard evaluation metrics
                    'iou': stats['iou'],
                    'loc_tp': stats['loc_tp'],
                    'loc_fp': stats['loc_fp'],
                    'loc_fn': stats['loc_fn'],
                    'loc_precision': stats['loc_precision'],
                    'loc_recall': stats['loc_recall'],
                    'loc_f1': stats['loc_f1'],
                    # LEGION evaluation metrics
                    'legion_iou': stats.get('legion_iou'),
                    'legion_miou': stats.get('legion_miou'),
                    'legion_iou_foreground': stats.get('legion_iou_foreground'),
                    'legion_iou_background': stats.get('legion_iou_background'),
                    'legion_pixel_f1': stats.get('legion_pixel_f1'),
                    'legion_pixel_precision': stats.get('legion_pixel_precision'),
                    'legion_pixel_recall': stats.get('legion_pixel_recall'),
                    # WSOL evaluation metrics
                    'wsol_iou': stats.get('wsol_iou'),
                    'prediction': prediction
                }
                if sample_result.get('iou', None) is None:
                    logger.info(f"Sample {i + 1} - Skipped (negative sample)")
                else:
                    logger.info(
                        f"Sample {i + 1} - IoU: {sample_result['iou']:.3f}, "
                        f"F1: {sample_result['loc_f1']:.3f} (P: {sample_result['loc_precision']:.3f}, "
                        f"R: {sample_result['loc_recall']:.3f}, TP/FP/FN: {sample_result['loc_tp']}/{sample_result['loc_fp']}/{sample_result['loc_fn']})"
                    )
            elif eval_type == 'explanation':
                sample_result = {
                    'image_path': str(image_path),
                    'rouge_l': stats['rouge_l'],
                    'css': stats['css'],
                    'prediction': prediction
                }
                logger.info(
                    f"Sample {i + 1} - ROUGE-L: {sample_result['rouge_l']:.3f}, "
                    f"CSS: {sample_result['css']:.3f}"
                )
            else:
                raise ValueError(f"Unsupported evaluation type: {eval_type}")

            # Store results
            results[i] = sample_result

        except Exception as e:
            logger.error(f"Error processing sample {i+1}: {e}")
            continue

    logger.info("Evaluation completed!")
    if results:
        # Initialize all variables
        binary_accuracy = 0.0
        f1_metrics = {}
        mean_iou = 0.0
        mean_rouge_l = 0.0
        mean_css = 0.0
        valid_loc_results = []
        mean_loc_f1 = 0.0
        mean_loc_precision = 0.0
        mean_loc_recall = 0.0
        total_loc_tp = 0
        total_loc_fp = 0
        total_loc_fn = 0
        
        if eval_type == 'binary':
            binary_accuracy = sum(r.get('binary_success', False) for _, r in results.items()) / total_samples
            # Compute F1 metrics
            f1_metrics = evaluator.compute_f1_metrics(results)
        elif eval_type == 'localization':
            # Filter out None values for SynArtifact negative samples
            valid_loc_results = [r for _, r in results.items() if r.get('iou') is not None]
            mean_iou = sum(r.get('iou', 0.0) for r in valid_loc_results) / len(valid_loc_results) if valid_loc_results else 0.0
            mean_loc_f1 = sum(r.get('loc_f1', 0.0) for r in valid_loc_results) / len(valid_loc_results) if valid_loc_results else 0.0
            mean_loc_precision = sum(r.get('loc_precision', 0.0) for r in valid_loc_results) / len(valid_loc_results) if valid_loc_results else 0.0
            mean_loc_recall = sum(r.get('loc_recall', 0.0) for r in valid_loc_results) / len(valid_loc_results) if valid_loc_results else 0.0
            total_loc_tp = sum(r.get('loc_tp', 0) for r in valid_loc_results if r.get('loc_tp') is not None)
            total_loc_fp = sum(r.get('loc_fp', 0) for r in valid_loc_results if r.get('loc_fp') is not None)
            total_loc_fn = sum(r.get('loc_fn', 0) for r in valid_loc_results if r.get('loc_fn') is not None)
            
            # LEGION evaluation metrics  
            legion_valid_results = [r for r in valid_loc_results if r.get('legion_iou') is not None]
            legion_mean_iou = sum(r.get('legion_iou', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_miou = sum(r.get('legion_miou', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_iou_fg = sum(r.get('legion_iou_foreground', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_iou_bg = sum(r.get('legion_iou_background', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_pixel_f1 = sum(r.get('legion_pixel_f1', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_pixel_precision = sum(r.get('legion_pixel_precision', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            legion_mean_pixel_recall = sum(r.get('legion_pixel_recall', 0.0) for r in legion_valid_results) / len(legion_valid_results) if legion_valid_results else 0.0
            
            # WSOL evaluation metrics
            wsol_valid_results = [r for r in valid_loc_results if r.get('wsol_iou') is not None]
            wsol_mean_iou = sum(r.get('wsol_iou', 0.0) for r in wsol_valid_results) / len(wsol_valid_results) if wsol_valid_results else 0.0
        elif eval_type == 'explanation':
            mean_rouge_l = sum(r.get('rouge_l', 0.0) for _, r in results.items()) / total_samples
            mean_css = sum(r.get('css', 0.0) for _, r in results.items()) / total_samples
        
        logger.info("=" * 60)
        logger.info("BATCH EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total samples processed: {total_samples}")
        if eval_type == 'binary':
            logger.info(f"Binary classification accuracy: {binary_accuracy:.3f}")
            logger.info("")
            logger.info("F1 METRICS:")
            logger.info(f"  TP: {f1_metrics.get('tp', 0)}, FP: {f1_metrics.get('fp', 0)}, FN: {f1_metrics.get('fn', 0)}, TN: {f1_metrics.get('tn', 0)}")
            logger.info(f"  Precision: {f1_metrics.get('precision', 0.0):.3f}")
            logger.info(f"  Recall: {f1_metrics.get('recall', 0.0):.3f}")
            logger.info(f"  F1-Score: {f1_metrics.get('f1_positive', 0.0):.3f}")
            logger.info(f"  Negative Precision: {f1_metrics['precision_negative']:.3f}")
            logger.info(f"  Negative Recall: {f1_metrics['recall_negative']:.3f}")
            logger.info(f"  Negative F1-Score: {f1_metrics['f1_negative']:.3f}")
            logger.info(f"  Macro F1: {f1_metrics['macro_f1']:.3f}")
            logger.info(f"  Accuracy: {f1_metrics.get('accuracy', 0.0):.3f}")
            logger.info("")
        elif eval_type == 'localization':
            valid_samples = len(valid_loc_results)
            logger.info("=" * 80)
            logger.info("COMPREHENSIVE LOCALIZATION EVALUATION RESULTS")
            logger.info("=" * 80)
            logger.info(f"Valid samples (positive samples): {valid_samples}")
            logger.info("")
            
            # Standard evaluation results
            logger.info("📊 STANDARD EVALUATION (Threshold-Independent Bbox Metrics):")
            logger.info(f"  Mean IoU: {mean_iou:.3f}")
            logger.info(f"  Mean F1: {mean_loc_f1:.3f}")
            logger.info(f"  Mean Precision: {mean_loc_precision:.3f}")
            logger.info(f"  Mean Recall: {mean_loc_recall:.3f}")
            logger.info(f"  Total TP/FP/FN: {total_loc_tp}/{total_loc_fp}/{total_loc_fn}")
            
            # Compute global F1 (across all samples)
            global_precision = total_loc_tp / (total_loc_tp + total_loc_fp) if (total_loc_tp + total_loc_fp) > 0 else 0.0
            global_recall = total_loc_tp / (total_loc_tp + total_loc_fn) if (total_loc_tp + total_loc_fn) > 0 else 0.0
            global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0
            logger.info(f"  Global Precision: {global_precision:.3f}")
            logger.info(f"  Global Recall: {global_recall:.3f}")
            logger.info(f"  Global F1: {global_f1:.3f}")
            logger.info("")
            
            # LEGION evaluation results
            legion_samples = len(legion_valid_results)
            logger.info("🎯 LEGION EVALUATION (Pixel-Level Segmentation Metrics):")
            logger.info(f"  Valid samples: {legion_samples}")
            logger.info(f"  Mean IoU (Legacy): {legion_mean_iou:.3f}")
            logger.info(f"  Mean IoU (mIoU): {legion_mean_miou:.3f}")
            logger.info(f"    - Foreground IoU: {legion_mean_iou_fg:.3f}")
            logger.info(f"    - Background IoU: {legion_mean_iou_bg:.3f}")
            logger.info(f"  Pixel F1 Score: {legion_mean_pixel_f1:.3f}")
            logger.info(f"  Pixel Precision: {legion_mean_pixel_precision:.3f}")
            logger.info(f"  Pixel Recall: {legion_mean_pixel_recall:.3f}")
            logger.info("")
            
            # WSOL evaluation results
            wsol_samples = len(wsol_valid_results)
            logger.info("🔄 WSOL EVALUATION (Threshold-Independent IoU):")
            logger.info(f"  Valid samples: {wsol_samples}")
            logger.info(f"  Mean IoU: {wsol_mean_iou:.3f}")
            logger.info("")
        elif eval_type == 'explanation':
            logger.info(f"Mean ROUGE-L (all samples): {mean_rouge_l:.3f}")
            logger.info(f"Mean CSS (all samples): {mean_css:.3f}")

    # Final GPU memory reporting for PAL model
    if config.get('model_type') == 'pal' and isinstance(model, PalEval):
        logger.info("📊 Final GPU Memory Usage:")
        memory_info = model.get_gpu_memory_info()
        for device, info in memory_info.items():
            logger.info(f"    {device} - Max Allocated: {info['max_allocated_gb']:.2f}GB, Current: {info['allocated_gb']:.2f}GB")
        
        # Final cache clearing
        model.clear_gpu_cache()

    if isinstance(model, GPTEval):
        try:
            logger.info(f"Total cost: {model.money_manager.total_cost}")
        except Exception:
            pass
    
    return results


def main_eval():
    """Main function for eval.py with pre-generated LEGION support"""
    
    parser = argparse.ArgumentParser(description="Evaluate artifact detection models with pre-generated responses")
    parser.add_argument('--model', required=True, help='Model to evaluate')
    parser.add_argument('--dataset', required=True, help='Dataset to evaluate on')
    parser.add_argument('--type', required=True, help='Evaluation type')
    parser.add_argument('--finetuned', action='store_true', help='Use finetuned model')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples to evaluate')
    
    args = parser.parse_args()
    
    # Setup logging first
    logger = setup_logging('eval_logs_refined', args.dataset, args.model, args.finetuned, args.type)
    
    # Set dataset paths
    dataset_paths = {
        'synthscars': "/data2/jhpark/image-artifacts/SynthScars/test",
        'synartifact': "/data2/jhpark/image-artifacts/SynArtifact/data",
        'loki': "/data2/jhpark/image-artifacts/loki",
        'richhf': "/data2/jhpark/image-artifacts/richhf-18k",
        "ours": "/data2/jhpark/image-artifacts/ours"
    }
    base_dir = dataset_paths.get(args.dataset)
    if base_dir is None:
        raise ValueError(f"No path configured for dataset: {args.dataset}")
    
    # Create config
    config = {
        'model_type': args.model,
        'dataset_type': args.dataset,
        'eval_type': args.type,
        'base_dir': base_dir,
        'use_finetuned': args.finetuned,
        'device': 'cuda'
    }
    
    print(f"🚀 Starting evaluation for {args.dataset.upper()} dataset")
    print(f"🤖 Model: {args.model}")
    print(f"🤖 Finetuned: {args.finetuned}")
    print(f"🗒️ Evaluating: {args.type}")
    
    try:
        patched_run_evaluation(config, args.max_samples, "eval")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


def main_legion_eval():
    """Main function for legion_eval.py with pre-generated LEGION support"""
    
    parser = argparse.ArgumentParser(description="LEGION-style evaluation with pre-generated responses")
    parser.add_argument('--model', required=True, help='Model to evaluate')
    parser.add_argument('--dataset', required=True, help='Dataset to evaluate on')
    parser.add_argument('--type', required=True, help='Evaluation type')
    parser.add_argument('--finetuned', action='store_true', help='Use finetuned model')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples to evaluate')
    
    args = parser.parse_args()
    
    # Setup logging first
    logger = setup_logging('eval_logs_refined', args.dataset, args.model, args.finetuned, args.type)
    
    # Set dataset paths
    dataset_paths = {
        'synthscars': "/data2/jhpark/image-artifacts/SynthScars/test",
        'synartifact': "/data2/jhpark/image-artifacts/SynArtifact/data",
        'loki': "/data2/jhpark/image-artifacts/loki",
        'richhf': "/data2/jhpark/image-artifacts/richhf-18k",
        "ours": "/data2/jhpark/image-artifacts/ours"
    }
    base_dir = dataset_paths.get(args.dataset)
    if base_dir is None:
        raise ValueError(f"No path configured for dataset: {args.dataset}")
    
    # Create config
    config = {
        'model_type': args.model,
        'dataset_type': args.dataset,
        'eval_type': args.type,
        'base_dir': base_dir,
        'use_finetuned': args.finetuned,
        'device': 'cuda:0' if os.system('nvidia-smi') == 0 else 'cpu'
    }
    
    print(f"🚀 Starting LEGION evaluation for {args.dataset.upper()} dataset")
    print(f"🤖 Model: {args.model}")
    print(f"🤖 Finetuned: {args.finetuned}")
    print(f"🗒️ Evaluating: {args.type}")
    
    try:
        patched_run_evaluation(config, args.max_samples, "legion_eval")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


def main_wsol_eval():
    """Main function for wsol_eval.py with pre-generated LEGION support"""
    
    parser = argparse.ArgumentParser(description="WSOL evaluation with pre-generated responses")
    parser.add_argument('--model', required=True, help='Model to evaluate')
    parser.add_argument('--dataset', required=True, help='Dataset to evaluate on')
    parser.add_argument('--type', required=True, help='Evaluation type')
    parser.add_argument('--finetuned', action='store_true', help='Use finetuned model')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples to evaluate')
    
    args = parser.parse_args()
    
    # Setup logging first
    logger = setup_logging('eval_logs_refined', args.dataset, args.model, args.finetuned, args.type)
    
    # Set dataset paths
    dataset_paths = {
        'synthscars': "/data2/jhpark/image-artifacts/SynthScars/test",
        'synartifact': "/data2/jhpark/image-artifacts/SynArtifact/data",
        'loki': "/data2/jhpark/image-artifacts/loki",
        'richhf': "/data2/jhpark/image-artifacts/richhf-18k",
        "ours": "/data2/jhpark/image-artifacts/ours"
    }
    base_dir = dataset_paths.get(args.dataset)
    if base_dir is None:
        raise ValueError(f"No path configured for dataset: {args.dataset}")
    
    # Create config
    config = {
        'model_type': args.model,
        'dataset_type': args.dataset,
        'eval_type': args.type,
        'base_dir': base_dir,
        'use_finetuned': args.finetuned,
        'device': 'cuda:0' if os.system('nvidia-smi') == 0 else 'cpu'
    }
    
    print(f"🚀 Starting WSOL evaluation for {args.dataset.upper()} dataset")
    print(f"🤖 Model: {args.model}")
    print(f"🤖 Finetuned: {args.finetuned}")
    print(f"🗒️ Evaluating: {args.type}")
    
    try:
        patched_run_evaluation(config, args.max_samples, "wsol_eval")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Determine which main function to run based on script name
    script_name = os.path.basename(sys.argv[0])
    if 'legion_eval' in script_name:
        main_legion_eval()
    elif 'wsol_eval' in script_name:
        main_wsol_eval()
    else:
        main_eval()
