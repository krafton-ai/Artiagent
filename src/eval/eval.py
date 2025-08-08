"""
Main evaluation script for artifact detection models.

This script evaluates VLM/MLLM models on their ability to detect
and describe visual artifacts in images across different datasets.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from PIL import Image
from pathlib import Path

from models import QwenEval, GPTEval, GeminiEval
from eval_utils import Evaluation, Visualizer, parse_tfrecord_file


def setup_logging(output_dir: str, dataset_type: str) -> logging.Logger:
    """
    Setup logging configuration with file and console handlers.
    
    Args:
        output_dir: Directory where logs will be saved
        dataset_type: Dataset name for log file naming
        
    Returns:
        Configured logger instance
    """
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'artifact_eval_{dataset_type}_{timestamp}.log'
    
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


class DatasetIterator:
    """
    Iterator for processing different artifact detection datasets.
    
    Supports SynthScars, SynArtifact, and LOKI datasets with
    unified interface for batch processing.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize dataset iterator.
        
        Args:
            config: Configuration dictionary with dataset settings
        """
        self.base_dir = Path(config['base_dir'])
        self.dataset_type = config['dataset_type']
        self.logger = logging.getLogger(__name__)
        
        # Initialize dataset-specific iterator
        if self.dataset_type == "synthscars":
            self._load_synthscars()
        elif self.dataset_type == "synartifact":
            self._load_synartifact()
        elif self.dataset_type == "loki":
            self._load_loki()
        elif self.dataset_type == "richhf":
            self._load_richhf()
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
            
        self.current_idx = 0
        self.total_samples = len(self.data)
        self.logger.info(f"Loaded {self.total_samples} samples from {self.dataset_type}")
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return self.total_samples
    
    def __iter__(self):
        """Make iterator iterable."""
        self.current_idx = 0
        return self
    
    def __next__(self) -> Tuple[Dict, Path]:
        """
        Get next sample from dataset.
        
        Returns:
            Tuple of (annotation_data, image_path)
            
        Raises:
            StopIteration: When all samples have been processed
        """
        if self.current_idx >= self.total_samples:
            raise StopIteration
        
        sample = self.data[self.current_idx]
        self.current_idx += 1
        
        return self._process_sample(sample)
    
    def get_sample(self, idx: int) -> Tuple[Dict, Path]:
        """
        Get specific sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (annotation_data, image_path)
        """
        if idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for {self.total_samples} samples")
        
        sample = self.data[idx]
        return self._process_sample(sample)
    
    def _process_sample(self, sample) -> Tuple[Dict, Path]:
        """Process a single sample based on dataset type."""
        if self.dataset_type == "synthscars":
            image_id, json_data = next(iter(sample.items()))
            image_dir = self.base_dir / "images"
            image_path = image_dir / json_data["img_file_name"]
            return json_data, image_path
            
        elif self.dataset_type == "synartifact":
            root_folder = sample.split('/')[0]
            image_id = Path(sample).stem
            
            image_path = self.base_dir / sample
            json_file = f"{root_folder}/annotation_json_artifacts_class/{image_id}.json"
            json_path = self.base_dir / json_file
            
            with open(json_path, "r") as f:
                json_data = json.load(f)
            
            return json_data, image_path
            
        elif self.dataset_type == "loki":
            json_data = sample
            image_path = self.base_dir / json_data["image_path"]
            return json_data, image_path

        elif self.dataset_type == "richhf":
            json_data = sample
            image_path = self.base_dir / json_data["filename"]
            return json_data, image_path
    
    def _load_synthscars(self):
        """Load SynthScars dataset."""
        json_path = self.base_dir / "annotations" / "test.json"
        with open(json_path, "rb") as f:
            self.data = json.load(f)
    
    def _load_synartifact(self):
        """Load SynArtifact dataset."""
        eval_set = self.base_dir / "eval.txt"
        self.data = []
        with open(eval_set, "r") as f:
            for line in f:
                self.data.append(line.strip())
    
    def _load_loki(self):
        """Load LOKI dataset."""
        json_path = self.base_dir / "open_ended_vqa.json"
        with open(json_path, "rb") as f:
            self.data = json.load(f)

    def _load_richhf(self):
        """Load RichHF-18K dataset from TFRecord file."""
        tfrecord_path = os.path.join(self.base_dir, "test.tfrecord")
        
        # Parse TFRecord file
        self.data = parse_tfrecord_file(tfrecord_path)

def create_model(config: Dict):
    """Create model instance based on configuration."""
    model_type = config.get('model_type', 'qwen')
    
    if model_type == 'qwen':
        return QwenEval(config)
    elif model_type == 'gpt':
        return GPTEval(config)
    elif model_type == 'gemini':
        return GeminiEval(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def run_evaluation(config: Dict, max_samples: Optional[int] = None, enable_visualization: bool = False):
    """
    Run evaluation on dataset.
    
    Args:
        config: Configuration dictionary
        max_samples: Maximum number of samples to evaluate (None for all)
        enable_visualization: Whether to save visualization results
    """
    logger = logging.getLogger(__name__)
    dataset_type = config['dataset_type']
    
    logger.info(f"Starting evaluation for {dataset_type} dataset")
    if config['use_finetuned']:
        logger.info("Using finetuned model")
    else:
        logger.info("Running zero-shot evaluation")
    
    # Initialize components
    logger.info("Initializing model and data iterator...")
    model = create_model(config)
    data_iterator = DatasetIterator(config)
    evaluator = Evaluation()

    if enable_visualization:
        viz_dir = os.path.join(config['log_dir'], 'visualizations')
        visualizer = Visualizer(viz_dir)
        logger.info(f"Visualization enabled. Outputs will be saved to: {viz_dir}")
    
    # Determine number of samples to process
    total_samples = len(data_iterator)
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)
    
    logger.info(f"Processing {total_samples} samples")

    results = {}
    
    # Process samples
    for i, (json_data, image_path) in enumerate(data_iterator):
        if max_samples is not None and i >= max_samples:
            break
            
        try:
            logger.info(f"Processing sample {i+1}/{total_samples}: {image_path}")
            
            # Load and process image
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
                
            image = Image.open(image_path).convert("RGB")
            if dataset_type == 'richhf':
                image = image.resize((512, 512), Image.LANCZOS)
            
            # Run model inference
            prediction = model.inference(image)
            
            # Evaluate results
            stats = evaluator.generate_statistics(
                dataset_type, json_data, prediction
            )
            sample_result = {
                'image_path': str(image_path),
                'binary_success': stats['binary_success'],
                'iou': stats['iou'],
                'rouge_l': stats['rouge_l'],
                'css': stats['css'],
                'classification': stats['classification'],
                'has_gt_artifacts': stats['has_gt_artifacts'],
                'has_pred_artifacts': stats['has_pred_artifacts'],
                'prediction': prediction
            }
            
            # Store results
            results[i] = sample_result

            if enable_visualization:
                try:
                    if dataset_type == 'loki':
                        viz_path = visualizer.visualize_loki(image, prediction, json_data, i)
                    elif dataset_type == 'synartifact':
                        viz_path = visualizer.visualize_synartifact(image, prediction, json_data, i)
                    elif dataset_type == 'synthscars':
                        viz_path = visualizer.visualize_synthscars(image, prediction, json_data, i)
                    elif dataset_type == 'richhf':
                        viz_path = visualizer.visualize_richhf(image, prediction, json_data, i)
                    else:
                        viz_path = None

                    if viz_path:
                        logger.info(f"Visualization saved to: {viz_path}")
            
                except Exception as e:
                    logger.warning(f"Visualization failed: {e}")
            
            logger.info(f"Sample {i+1} - Binary: {stats['binary_success']}, IoU: {stats['iou']:.3f}, "
                       f"ROUGE-L: {stats['rouge_l']:.3f}, CSS: {stats['css']:.3f}")
                       
        except Exception as e:
            logger.error(f"Error processing sample {i+1}: {e}")
            continue

    logger.info("Evaluation completed!")
    if results:
        binary_accuracy = sum(r['binary_success'] for _, r in results.items()) / total_samples
        mean_iou = sum(r['iou'] for _, r in results.items()) / total_samples
        mean_rouge_l = sum(r['rouge_l'] for _, r in results.items()) / total_samples
        mean_css = sum(r['css'] for _, r in results.items()) / total_samples

        f1_metrics = evaluator.compute_f1_metrics(results)

        logger.info(f"Summary Results:")
        logger.info(f"  Binary Classification Accuracy: {binary_accuracy:.3f}")
        logger.info(f"Mean IoU (all samples): {mean_iou:.3f}")
        logger.info(f"Mean ROUGE-L (all samples): {mean_rouge_l:.3f}")
        logger.info(f"Mean CSS (all samples): {mean_css:.3f}")
        logger.info("")
        logger.info("F1 METRICS:")
        logger.info(f"  TP: {f1_metrics['tp']}, FP: {f1_metrics['fp']}, FN: {f1_metrics['fn']}, TN: {f1_metrics['tn']}")
        logger.info(f"  Precision: {f1_metrics['precision']:.3f}")
        logger.info(f"  Recall: {f1_metrics['recall']:.3f}")
        logger.info(f"  F1-Score: {f1_metrics['f1_score']:.3f}")
        logger.info(f"  Accuracy: {f1_metrics['accuracy']:.3f}")
        logger.info("")
        logger.info("METRICS FOR TRUE POSITIVE CASES ONLY:")
        logger.info(f"  Mean TP IoU: {f1_metrics['mean_tp_iou']:.3f}")
        logger.info(f"  Mean TP ROUGE-L: {f1_metrics['mean_tp_rouge']:.3f}")
        logger.info(f"  Mean TP CSS: {f1_metrics['mean_tp_css']:.3f}")

    if config['model_type'] == 'gpt':
        logger.info(f"Total cost: {model.money_manager.total_cost}")
    
    return results

def run_batch_evaluation(config: Dict, max_samples: Optional[int] = None, enable_visualization: bool = False):
    """
    Run evaluation on multiple images with optional visualization.
    
    Args:
        config: Configuration dictionary
        max_samples: Maximum number of samples to process
        enable_visualization: Whether to save visualization results
    """
    logger = logging.getLogger(__name__)
    dataset_type = config['dataset_type']

    # Setup logging
    logger.info(f"Starting evaluation for {dataset_type} dataset")
    if config['use_finetuned']:
        logger.info("Using finetuned model")
    else:
        logger.info("Running zero-shot evaluation")
    
    logger.info(f"Starting batch evaluation for {dataset_type} dataset")
    if max_samples:
        logger.info(f"Processing {max_samples} samples")
    
    # Initialize components
    logger.info("Initializing model and data iterator...")
    model = QwenEval(config)
    data_iterator = DatasetIterator(config)
    evaluator = Evaluation()
    
    if enable_visualization:
        viz_dir = os.path.join(log_dir, 'visualizations')
        visualizer = Visualizer(viz_dir)
        logger.info(f"Visualization enabled. Outputs will be saved to: {viz_dir}")
    
    total_samples = len(data_iterator)
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)
    
    results = {}
    processed = 0

    # Determine batch size: default to 2 if not provided in config
    target_batch_size: int = int(config.get('batch_size', 2) or 2)
    current_batch_size: int = target_batch_size

    try:
        while True:
            if max_samples and processed >= max_samples:
                break
            # Collect a batch of samples
            batch_json_data: List[Dict] = []
            batch_image_paths: List[str] = []
            batch_images: List[Image.Image] = []
            while len(batch_images) < current_batch_size:
                try:
                    json_data, image_path = next(data_iterator)
                except StopIteration:
                    break
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    logger.warning(f"Failed to load image {image_path}: {e}")
                    continue
                batch_json_data.append(json_data)
                batch_image_paths.append(image_path)
                batch_images.append(image)
                if max_samples and (processed + len(batch_images)) >= max_samples:
                    break
            if not batch_images:
                # No more data
                break
            logger.info(
                f"Processing batch starting at index {processed + 1} with size {len(batch_images)}"
            )
            # Run batched inference with OOM fallback
            try:
                batch_results = model.inference_batch(batch_images)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and len(batch_images) > 1:
                    logger.warning("OOM during batched inference. Falling back to per-sample inference for this batch.")
                    # Reduce future batch size to be more conservative
                    current_batch_size = max(1, current_batch_size // 2)
                    batch_results = []
                    for img in batch_images:
                        try:
                            batch_results.append(model.inference(img))
                        except Exception as inner_e:
                            logger.error(f"Per-sample inference failed: {inner_e}")
                            batch_results.append({
                                'number_of_artifacts': 0,
                                'artifacts': [],
                                'error': str(inner_e)
                            })
                else:
                    raise
            # Evaluate each item in the batch
            for idx, (json_data, image_path, image, result) in enumerate(
                zip(batch_json_data, batch_image_paths, batch_images, batch_results)
            ):
                stats = evaluator.generate_statistics(
                    dataset_type, json_data, result
                )
                sample_result = {
                    'image_path': str(image_path),
                    'binary_success': stats['binary_success'],
                    'iou': stats['iou'],
                    'rouge_l': stats['rouge_l'],
                    'css': stats['css'],
                    'classification': stats['classification'],
                    'has_gt_artifacts': stats['has_gt_artifacts'],
                    'has_pred_artifacts': stats['has_pred_artifacts'],
                    'prediction': result,
                }
                # Visualization per item if enabled
                if enable_visualization and visualizer is not None:
                    try:
                        if dataset_type == 'loki':
                            viz_path = visualizer.visualize_loki(image, result, json_data, processed + idx)
                        elif dataset_type == 'synartifact':
                            viz_path = visualizer.visualize_synartifact(image, result, json_data, processed + idx)
                        elif dataset_type == 'synthscars':
                            viz_path = visualizer.visualize_synthscars(image, result, json_data, processed + idx)
                        elif dataset_type == 'richhf':
                            viz_path = visualizer.visualize_richhf(image, result, json_data, processed + idx)
                        else:
                            viz_path = None
                        sample_result['visualization_path'] = viz_path
                        if viz_path:
                            logger.info(
                                f"Sample {processed + idx + 1} visualization saved to: {viz_path}"
                            )
                    except Exception as viz_e:
                        logger.warning(
                            f"Visualization failed for sample {processed + idx + 1}: {viz_e}"
                        )
                        sample_result['visualization_path'] = None
                results[processed + idx] = sample_result
                logger.info(
                    f"Sample {processed + idx + 1} - Binary: {sample_result['binary_success']}, "
                    f"IoU: {sample_result['iou']:.3f}, ROUGE-L: {sample_result['rouge_l']:.3f}, "
                    f"CSS: {sample_result['css']:.3f}"
                )
            processed += len(batch_images)
            
    except StopIteration:
        logger.info("Reached end of dataset")
    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
    
    # Compute summary statistics
    if results:
        total_samples = len(results)
        binary_accuracy = sum(r['binary_success'] for _, r in results.items()) / total_samples
        mean_iou = sum(r['iou'] for _, r in results.items()) / total_samples
        mean_rouge_l = sum(r['rouge_l'] for _, r in results.items()) / total_samples
        mean_css = sum(r['css'] for _, r in results.items()) / total_samples
        
        # Compute F1 metrics
        f1_metrics = evaluator.compute_f1_metrics(results)
        
        logger.info("=" * 60)
        logger.info("BATCH EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total samples processed: {total_samples}")
        logger.info(f"Binary classification accuracy: {binary_accuracy:.3f}")
        logger.info(f"Mean IoU (all samples): {mean_iou:.3f}")
        logger.info(f"Mean ROUGE-L (all samples): {mean_rouge_l:.3f}")
        logger.info(f"Mean CSS (all samples): {mean_css:.3f}")
        logger.info("")
        logger.info("F1 METRICS:")
        logger.info(f"  TP: {f1_metrics['tp']}, FP: {f1_metrics['fp']}, FN: {f1_metrics['fn']}, TN: {f1_metrics['tn']}")
        logger.info(f"  Precision: {f1_metrics['precision']:.3f}")
        logger.info(f"  Recall: {f1_metrics['recall']:.3f}")
        logger.info(f"  F1-Score: {f1_metrics['f1_score']:.3f}")
        logger.info(f"  Accuracy: {f1_metrics['accuracy']:.3f}")
        logger.info("")
        logger.info("METRICS FOR TRUE POSITIVE CASES ONLY:")
        logger.info(f"  Mean TP IoU: {f1_metrics['mean_tp_iou']:.3f}")
        logger.info(f"  Mean TP ROUGE-L: {f1_metrics['mean_tp_rouge']:.3f}")
        logger.info(f"  Mean TP CSS: {f1_metrics['mean_tp_css']:.3f}")
        
        # Save summary report if visualization was enabled
        if enable_visualization and results:
            report_path = visualizer.save_summary_report(results, dataset_type)
            logger.info(f"Visualization report saved to: {report_path}")
    
    return results

def main():
    """Main function for model evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate VLM/MLLM models on artifact detection tasks'
    )
    parser.add_argument('--model', type=str, choices=['qwen', 'gpt', 'gemini'], 
                       default='qwen', help='Model type to evaluate (default: qwen)')
    parser.add_argument('--dataset', type=str, 
                       choices=['synthscars', 'synartifact', 'loki', 'richhf'], 
                       default='loki', help='Dataset to evaluate on (default: loki)')
    parser.add_argument('--use-finetuned', action='store_true',
                       help='Use finetuned model instead of base model')
    parser.add_argument('--device', type=str, default="cuda:0",
                       help='Device for inference (default: cuda:0)')
    parser.add_argument('--log-dir', type=str, default='eval_logs',
                       help='Directory for logs (default: eval_logs)')
    parser.add_argument('--output-dir', type=str, default='eval_results',
                       help='Directory for results (default: eval_results)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (default: all)')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization of results (default: False)')
    parser.add_argument('--base-dir', type=str, default=None,
                       help='Custom base directory for dataset')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batched inference size (default: 2)')
                       
    args = parser.parse_args()
    
    # Set dataset paths if not provided
    if args.base_dir is None:
        dataset_paths = {
            'synthscars': "/home/jovyan/image-artifacts/data/SynthScars/test",
            'synartifact': "/home/jovyan/image-artifacts/data/SynArtifact/data",
            'loki': "/home/jovyan/image-artifacts/data/loki",
            'richhf': "/home/jovyan/image-artifacts/data/richhf-18k"
        }
        base_dir = dataset_paths.get(args.dataset)
        if base_dir is None:
            raise ValueError(f"No default path for dataset: {args.dataset}")
    else:
        base_dir = args.base_dir
    
    # Setup configuration
    config = {
        'model_type': args.model,
        'dataset_type': args.dataset,
        'base_dir': base_dir,
        'log_dir': args.log_dir,
        'use_finetuned': args.use_finetuned,
        'device': args.device,
        'batch_size': args.batch_size
    }
    
    # Setup logging
    logger = setup_logging(args.log_dir, args.dataset)
    
    print(f"🚀 Starting evaluation for {args.dataset.upper()} dataset")
    print(f"🤖 Model: {args.model}")
    print(f"📁 Dataset path: {base_dir}")
    print(f"🔧 Device: {args.device}")
    
    try:
        # Run evaluation
        if args.batch_size > 1:
            results = run_batch_evaluation(config, args.max_samples, args.visualize)
        else:
            results = run_evaluation(config, args.max_samples, args.visualize)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"results_{args.dataset}_{args.model}_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Evaluation completed! Results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user.")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()