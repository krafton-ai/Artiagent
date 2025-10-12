"""
Batch Evaluation for Finetuned VLM using TRUE Batch Processing (Collator Method)

This implementation mimics exactly how LLaMA-Factory's MultiModalDataCollatorForSeq2Seq
batches images during training to enable TRUE batch inference during evaluation.

Key insight: The collator concatenates pixel_values from all images and uses image_grid_thw
to track which patches belong to which image, allowing batching of different-sized images!
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "train" / "LLaMA-Factory" / "src"))

from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import ModelArguments, DataArguments, EvaluationArguments, get_eval_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data.mm_plugin import IMAGE_PLACEHOLDER

from eval_finetuned_llamafactory import (
    DatasetIterator,
    process_finetuned_output,
    setup_logging,
)
from eval_utils import create_prompt, Evaluation

logger = logging.getLogger(__name__)


class FinetunedModelEvaluatorTrueBatch:
    """
    Evaluator that implements TRUE batch processing by mimicking training collator.
    
    This follows the exact pattern from MultiModalDataCollatorForSeq2Seq:
    1. Process each sample individually (like DataLoader does)
    2. Flatten images and track counts with imglens
    3. Call get_mm_inputs with all images (like collator does) 
    4. Pad input_ids and run batched generation
    """
    
    def __init__(self, exp_dir: str, device: str = "cuda:0"):
        self.exp_dir = exp_dir
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.template = None
        
    def load_model(self):
        """Load model using LLaMA-Factory's evaluation framework."""
        logger.info("Loading model using LLaMA-Factory evaluation framework...")
        
        # Configure arguments
        args_dict = {
            "model_name_or_path": self.exp_dir,
            "template": "qwen2_vl",
            "task": "mmlu",  # Placeholder required by get_eval_args
            "infer_backend": "huggingface",
            "infer_dtype": "bfloat16",
        }
        
        # Convert to args list
        args_list = []
        for key, value in args_dict.items():
            args_list.append(f"--{key}")
            args_list.append(str(value))
        
        # Parse arguments
        model_args, data_args, eval_args, finetuning_args = get_eval_args(args_list)
        
        # Load tokenizer and processor
        tokenizer_module = load_tokenizer(model_args)
        self.tokenizer = tokenizer_module["tokenizer"]
        self.processor = tokenizer_module.get("processor")
        
        # Get template
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)
        
        # Load model
        self.model = load_model(self.tokenizer, model_args, finetuning_args)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def inference_batch(self, images: List[Image.Image], prompt: str) -> List[str]:
        """
        Perform TRUE batch inference mimicking the training collator approach.
        
        Args:
            images: List of PIL images
            prompt: Text prompt to use for all images
            
        Returns:
            List of model responses (one per image)
        """
        batch_size = len(images)
        
        # Step 1: Process each sample individually (like DataLoader)
        # This creates input_ids with the correct number of image tokens per sample
        batch_input_ids = []
        for image in images:
            # Create messages
            messages = [
                {
                    "role": "user",
                    "content": f"{IMAGE_PLACEHOLDER}{prompt}"
                }
            ]
            
            # Process messages (expands image tokens based on resolution)
            if hasattr(self.template, 'mm_plugin') and self.template.mm_plugin:
                processed_messages = self.template.mm_plugin.process_messages(
                    messages, [image], [], [], self.processor
                )
            else:
                processed_messages = messages
            
            # Encode messages
            paired_messages = processed_messages + [{"role": "assistant", "content": ""}]
            prompt_ids, _ = self.template.encode_oneturn(self.tokenizer, paired_messages)
            
            # Process token IDs
            if hasattr(self.template, 'mm_plugin') and self.template.mm_plugin:
                prompt_ids, _ = self.template.mm_plugin.process_token_ids(
                    prompt_ids,
                    None,
                    [image],
                    [],
                    [],
                    self.tokenizer,
                    self.processor,
                )
            
            batch_input_ids.append(prompt_ids)
        
        # Step 2: Flatten images and track counts (exactly like collator at line 115-121)
        batch_images = images  # Flat list of all images
        batch_imglens = [1] * batch_size  # One image per sample
        batch_vidlens = [0] * batch_size
        batch_audlens = [0] * batch_size
        
        # Step 3: Get multimodal inputs (exactly like collator at line 168-177)
        # This returns CONCATENATED pixel_values and BATCHED image_grid_thw
        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            images=batch_images,
            videos=[],
            audios=[],
            imglens=batch_imglens,
            vidlens=batch_vidlens,
            audlens=batch_audlens,
            batch_ids=batch_input_ids,
            processor=self.processor
        )
        
        # Step 4: Pad input_ids to create batch (like collator line 183)
        batch = self.tokenizer.pad(
            {"input_ids": batch_input_ids},
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Step 5: Prepare generation arguments (like collator line 225)
        gen_kwargs = {
            **batch,  # input_ids and attention_mask
            "max_new_tokens": 512,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Add multimodal inputs (like collator line 225)
        for key, value in mm_inputs.items():
            if hasattr(value, 'to'):
                gen_kwargs[key] = value.to(self.model.device)
            else:
                gen_kwargs[key] = value
        
        # Step 6: Run TRUE batch inference - single forward pass!
        with torch.inference_mode():
            outputs = self.model.generate(**gen_kwargs)
        
        # Step 7: Decode outputs individually
        results = []
        for i in range(batch_size):
            input_len = batch["attention_mask"][i].sum().item()
            response_ids = outputs[i, input_len:]
            response = self.tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            results.append(response)
        
        logger.info(f"✅ TRUE batch processing (collator method) - {batch_size} images in ONE forward pass!")
        
        return results
    
    def inference(self, image: Image.Image, prompt: str) -> str:
        """Single image inference (calls batch inference with size 1)."""
        return self.inference_batch([image], prompt)[0]


def run_batch_evaluation_true_batch(args):
    """Run batch evaluation using true batch processing (collator method)."""
    
    # Setup logging
    exp_name = Path(args.exp_dir).name
    log_dir = Path(__file__).parent / "eval_logs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = setup_logging(
        str(log_dir),
        args.dataset,
        exp_name,
        args.type
    )
    
    logger.info("🚀 Starting TRUE batch evaluation (collator method)")
    logger.info(f"📁 Experiment directory: {args.exp_dir}")
    logger.info(f"📊 Dataset: {args.dataset.upper()}")
    logger.info(f"🗒️ Evaluation type: {args.type}")
    logger.info(f"🔧 Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Initialize model and data iterator
    logger.info("Initializing model and data iterator...")
    evaluator_model = FinetunedModelEvaluatorTrueBatch(args.exp_dir, args.device)
    evaluator_model.load_model()
    
    # Setup dataset iterator
    config = {
        'dataset_type': args.dataset,  # Use dataset_type for consistency
        'eval_type': args.type,
        'data_path': args.dataset_path,
        'base_dir': args.dataset_path,
    }
    data_iterator = DatasetIterator(config)
    
    # Limit samples if specified
    if args.max_samples:
        # DatasetIterator doesn't support max_samples directly, we'll handle it in the loop
        pass
    
    # Setup evaluation metrics
    evaluator = Evaluation()
    
    # Create query
    input_query = create_prompt(args.type)
    logger.info(f"Input query: {input_query[:200]}...")
    
    # Batch evaluation
    all_rouge_l_scores = []
    all_css_scores = []
    all_results = []
    
    batch_images = []
    batch_metadata = []
    sample_idx = 1
    total_processed = 0
    
    # Determine total samples for progress bar
    total_samples = args.max_samples if args.max_samples else len(data_iterator)
    
    # Create progress bar for the entire evaluation
    pbar = tqdm(total=total_samples, desc="Evaluating samples", unit="sample")
    
    for gt, image_path in data_iterator:  # DatasetIterator returns (json_data, image_path)
        # Check max samples
        if args.max_samples and total_processed >= args.max_samples:
            break
        # Load image
        image = Image.open(image_path).convert('RGB')
        batch_images.append(image)
        batch_metadata.append((image_path, gt))
        
        # Process batch when full or at end
        if len(batch_images) == args.batch_size or sample_idx == len(data_iterator):
            logger.info(f"Processing batch starting at index {sample_idx} with size {len(batch_images)}")
            
            # Batch inference
            try:
                batch_raw_outputs = evaluator_model.inference_batch(batch_images, input_query)
                
                # Process each output in the batch
                for i, (raw_output, (img_path, gt)) in enumerate(zip(batch_raw_outputs, batch_metadata)):
                    # Process output
                    batch_predictions = process_finetuned_output(raw_output, args.type)
                    
                    # Get image for size info
                    image = batch_images[i]
                    
                    # Calculate stats using the correct signature
                    stats = evaluator.generate_statistics(
                        args.dataset,  # dataset_type
                        args.type,     # eval_type
                        gt,            # json_data (ground truth)
                        batch_predictions,  # result (prediction)
                        image_size=image.size
                    )
                    
                    rouge_l = stats.get('rouge_l', 0.0)
                    css = stats.get('css', 0.0)
                    
                    all_rouge_l_scores.append(rouge_l)
                    all_css_scores.append(css)
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'ROUGE-L': f'{rouge_l:.3f}',
                        'CSS': f'{css:.3f}',
                        'Avg_ROUGE': f'{sum(all_rouge_l_scores)/len(all_rouge_l_scores):.3f}',
                        'Avg_CSS': f'{sum(all_css_scores)/len(all_css_scores):.3f}'
                    })
                    
                    logger.info(f"Sample {sample_idx + i - len(batch_images) + 1} - ROUGE-L: {rouge_l:.3f}, CSS: {css:.3f}")
                    
                    # Store result
                    result_entry = {
                        'image_path': str(img_path),
                        'ground_truth': gt,
                        'prediction': batch_predictions,
                        'rouge_l': rouge_l,
                        'css': css
                    }
                    all_results.append(result_entry)
                    total_processed += 1
                
            except Exception as e:
                logger.error(f"Error during batch evaluation: {e}")
                import traceback
                traceback.print_exc()
            
            # Reset batch
            batch_images = []
            batch_metadata = []
        
        sample_idx += 1
    
    # Close progress bar
    pbar.close()
    
    # Calculate final metrics
    mean_rouge_l = sum(all_rouge_l_scores) / len(all_rouge_l_scores) if all_rouge_l_scores else 0.0
    mean_css = sum(all_css_scores) / len(all_css_scores) if all_css_scores else 0.0
    
    logger.info("Batch evaluation completed!")
    logger.info("=" * 60)
    logger.info(f"{args.type.upper()} EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total samples processed: {len(all_rouge_l_scores)}")
    logger.info(f"Mean ROUGE-L: {mean_rouge_l:.3f}")
    logger.info(f"Mean CSS: {mean_css:.3f}")
    
    # Save results
    results_dir = Path(__file__).parent / "eval_results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"finetuned_results_true_batch_{args.dataset}_{exp_name}_{args.type}_{timestamp}.json"
    
    final_results = {
        'config': {
            'exp_dir': args.exp_dir,
            'dataset': args.dataset,
            'eval_type': args.type,
            'batch_size': args.batch_size,
            'max_samples': args.max_samples,
            'method': 'true_batch_collator_method'
        },
        'metrics': {
            'mean_rouge_l': mean_rouge_l,
            'mean_css': mean_css,
        },
        'results': all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✅ Evaluation completed! Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="TRUE Batch Evaluation (Collator Method)")
    parser.add_argument("--exp-dir", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--dataset", type=str, default="ours", choices=["ours", "t2i"], help="Dataset to evaluate on")
    parser.add_argument("--type", type=str, default="explanation", choices=["explanation", "classification", "all"], help="Type of evaluation")
    parser.add_argument("--dataset-path", type=str, default="/data2/jhpark/image-artifacts/data/eval", help="Path to dataset")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run evaluation on")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to evaluate")
    
    args = parser.parse_args()
    
    run_batch_evaluation_true_batch(args)


if __name__ == "__main__":
    main()

