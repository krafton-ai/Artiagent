#!/usr/bin/env python3
"""
Real Image Description Generator for ArtiAgent Instances

This script processes directories containing ArtiAgent instances and generates
straightforward descriptions of real (clean) images without any artifact context.

The script:
1. Traverses directories containing ArtiAgent instances
2. Loads real images and existing metadata
3. Generates descriptions using the real_image_description function
4. Saves results back to metadata.json files (adds 'real_description' field)
5. Tracks API costs and provides progress feedback

Usage:
    python real_description.py --data_dir <path> [--max_workers <num>]
    
Example:
    python real_description.py --data_dir ~/image-artifacts/data/train/vanilla_fireflow_comparison/person/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import logging
from datetime import datetime
import time

import numpy as np
from PIL import Image

# Import functions from prompts
from pipeline.prompts import real_image_description, MoneyManager
from openai import OpenAI


def setup_logging(output_dir: Path):
    """Setup logging configuration"""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'real_description_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class RealDescriptionGenerator:
    """Generate descriptions for real images in ArtiAgent instances"""
    
    def __init__(self, data_dir: str, max_workers: int = 8, max_retries: int = 3):
        """
        Initialize the real description generator.
        
        Args:
            data_dir: Root directory containing ArtiAgent instances
            max_workers: Maximum number of threads for parallel processing
            max_retries: Maximum number of retry attempts for failed instances
        """
        self.data_dir = Path(data_dir)
        self.max_workers = max_workers
        self.max_retries = max_retries
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        # Initialize OpenAI client and money manager
        self.client = OpenAI()
        self.money_manager = MoneyManager(model="gpt-4o")
        
        # Thread safety
        self.results_lock = threading.Lock()
        self.processed_count = 0
        self.failed_count = 0
        
        # Setup logging
        self.logger = setup_logging(self.data_dir)
    
    def find_artiagent_instances(self) -> List[Path]:
        """
        Find all ArtiAgent instance directories.
        
        An instance directory should contain:
        - metadata.json
        - real_image.png
        
        Returns:
            List of paths to valid instance directories
        """
        instance_dirs = []
        
        # Traverse the directory tree
        for item in self.data_dir.rglob('*'):
            if item.is_dir():
                # Check if this directory contains required files
                metadata_path = item / 'metadata.json'
                real_image_path = item / 'real_image.png'
                
                if metadata_path.exists() and real_image_path.exists():
                    instance_dirs.append(item)
        
        self.logger.info(f"Found {len(instance_dirs)} ArtiAgent instances")
        return sorted(instance_dirs)
    
    def process_instance(self, instance_dir: Path, retry_count: int = 0) -> Dict:
        """
        Process a single ArtiAgent instance and generate real image description.
        
        Args:
            instance_dir: Path to instance directory
            retry_count: Current retry attempt number
            
        Returns:
            Dict with processing results
        """
        # Create thread-specific client and money manager
        thread_client = OpenAI()
        thread_money_manager = MoneyManager(model="gpt-4o")
        
        instance_id = instance_dir.name
        
        try:
            # Load metadata
            metadata_path = instance_dir / 'metadata.json'
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if real_description already exists
            if 'real_description' in metadata and metadata['real_description']:
                self.logger.info(f"Instance {instance_id}: real_description already exists, skipping")
                return {
                    'success': True,
                    'instance_id': instance_id,
                    'cost': 0.0,
                    'skipped': True
                }
            
            # Load real image
            real_image = Image.open(instance_dir / 'real_image.png')
            
            # Generate real image description
            real_desc = real_image_description(
                thread_client,
                real_image,
                thread_money_manager
            )
            
            if not real_desc:
                error_msg = 'Failed to generate description'
                self.logger.error(f"Instance {instance_id}: {error_msg} (attempt {retry_count + 1}/{self.max_retries + 1})")
                return {
                    'success': False,
                    'instance_id': instance_id,
                    'error': error_msg,
                    'cost': thread_money_manager.total_cost,
                    'retry_count': retry_count
                }
            
            # Update metadata with real description
            metadata['real_description'] = real_desc
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Instance {instance_id}: Successfully generated real description")
            self.logger.info(f"  Description: {real_desc[:100]}...")
            
            return {
                'success': True,
                'instance_id': instance_id,
                'real_description': real_desc,
                'cost': thread_money_manager.total_cost,
                'skipped': False,
                'retry_count': retry_count
            }
            
        except Exception as e:
            self.logger.error(f"Instance {instance_id}: Error - {str(e)} (attempt {retry_count + 1}/{self.max_retries + 1})")
            return {
                'success': False,
                'instance_id': instance_id,
                'error': str(e),
                'cost': 0.0,
                'retry_count': retry_count
            }
    
    def run(self):
        """Run the real description generation pipeline with retry logic"""
        self.logger.info(f"Starting real description generation for {self.data_dir}")
        
        # Find all ArtiAgent instances
        instance_dirs = self.find_artiagent_instances()
        
        if not instance_dirs:
            self.logger.warning("No ArtiAgent instances found!")
            return
        
        total_instances = len(instance_dirs)
        self.logger.info(f"Processing {total_instances} instances with {self.max_workers} threads...")
        self.logger.info(f"Max retries per instance: {self.max_retries}")
        
        # Process instances in parallel with retry logic
        passed_experiments = 0
        skipped_count = 0
        total_phase_cost = 0.0
        failed_instances = []
        
        # Initial processing
        remaining_instances = list(instance_dirs)
        
        for retry_round in range(self.max_retries + 1):
            if not remaining_instances:
                break
            
            if retry_round > 0:
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"RETRY ROUND {retry_round}/{self.max_retries}")
                self.logger.info(f"Retrying {len(remaining_instances)} failed instances...")
                self.logger.info(f"{'='*80}\n")
                # Add a small delay before retrying
                time.sleep(2)
            
            current_round_failures = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks for current round
                futures = {
                    executor.submit(self.process_instance, instance_dir, retry_round): instance_dir
                    for instance_dir in remaining_instances
                }
                
                # Initialize progress bar
                progress_bar = tqdm(
                    total=len(futures),
                    desc=f"Round {retry_round + 1}" if retry_round > 0 else "Processing",
                    unit="inst",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
                )
                
                # Process results as they complete
                for future in as_completed(futures):
                    instance_dir = futures[future]
                    
                    try:
                        result = future.result()
                        total_phase_cost += result.get('cost', 0.0)
                        
                        # Update counts
                        with self.results_lock:
                            if result['success']:
                                self.processed_count += 1
                                if result.get('skipped', False):
                                    skipped_count += 1
                            else:
                                # Track for potential retry
                                current_round_failures.append(instance_dir)
                        
                        # Update progress bar
                        progress_bar.set_postfix({
                            '✓': self.processed_count,
                            '⊘': skipped_count,
                            '✗': len(current_round_failures)
                        })
                        progress_bar.update(1)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {instance_dir}: {str(e)}")
                        current_round_failures.append(instance_dir)
                        progress_bar.update(1)
                
                progress_bar.close()
            
            # Update remaining instances for next retry round
            remaining_instances = current_round_failures
            
            if retry_round < self.max_retries and remaining_instances:
                self.logger.info(f"Round {retry_round + 1} complete: {len(remaining_instances)} instances still failing")
            elif retry_round == self.max_retries and remaining_instances:
                self.logger.warning(f"Max retries reached. {len(remaining_instances)} instances permanently failed.")
                failed_instances = remaining_instances
        
        # Update total cost
        self.money_manager.total_cost += total_phase_cost
        
        # Calculate final counts
        passed_experiments = self.processed_count - skipped_count
        self.failed_count = len(failed_instances)
        
        # Print final summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"Total instances: {total_instances}")
        print(f"Successfully processed: {passed_experiments}")
        print(f"Skipped (already had real_description): {skipped_count}")
        print(f"Failed (after {self.max_retries} retries): {self.failed_count}")
        print(f"Total cost: ${self.money_manager.total_cost:.4f}")
        
        # Save summary to file
        summary_file = self.data_dir / "real_description_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Real Image Description Generation Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total instances: {total_instances}\n")
            f.write(f"Successfully processed: {passed_experiments}\n")
            f.write(f"Skipped (already had real_description): {skipped_count}\n")
            f.write(f"Failed (after {self.max_retries} retries): {self.failed_count}\n")
            f.write(f"Max retries per instance: {self.max_retries}\n")
            f.write(f"Total cost: ${self.money_manager.total_cost:.4f}\n")
            
            if failed_instances:
                f.write(f"\nPermanently Failed Instances ({len(failed_instances)}):\n")
                for instance_dir in failed_instances:
                    f.write(f"  - {instance_dir.name}\n")
        
        self.logger.info(f"Summary saved to: {summary_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Generate descriptions for real images in ArtiAgent instances'
    )
    parser.add_argument(
        '--data-dir',
        required=True,
        help='Root directory containing ArtiAgent instances'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=8,
        help='Maximum number of threads for parallel processing (default: 8)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum number of retry attempts for failed instances (default: 3)'
    )
    
    args = parser.parse_args()
    
    print("Real Image Description Generator")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Max workers: {args.max_workers}")
    print(f"Max retries: {args.max_retries}")
    print()
    
    # Initialize and run the generator
    generator = RealDescriptionGenerator(
        data_dir=args.data_dir,
        max_workers=args.max_workers,
        max_retries=args.max_retries
    )
    
    generator.run()
    
    print("\n" + "=" * 80)
    print("Real description generation completed!")


if __name__ == "__main__":
    main()
