import os
import json
import glob
import logging
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from prompts import MoneyManager, artifact_explanation

# -----------------------------
# Configure Thread-Safe Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s"
)

# -----------------------------
# Worker Function for Threads
# -----------------------------
def process_single_image_dir(image_dir, client):
    metadata_path = os.path.join(image_dir, "metadata.json")
    original_image_path = os.path.join(image_dir, "original_image.png")
    artifact_image_path = os.path.join(image_dir, "artifact_image.png")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load images once for all artifacts in this directory
    original_image = Image.open(original_image_path)
    artifact_image = Image.open(artifact_image_path)
    
    # Create one money manager for all artifacts in this directory
    money_manager = MoneyManager(model="gpt-4o")
    
    artifacts_processed = 0
    artifacts_failed = 0

    for artifact in metadata:
        try:
            target_bbox = artifact['target_bbox']
            artifact_type = artifact['artifact_type']
            entity = artifact['entity']
            subentity = artifact['subentity']

            # Create copies of images with bounding boxes for this artifact
            original_with_bbox = original_image.copy()
            artifact_with_bbox = artifact_image.copy()
            
            draw = ImageDraw.Draw(original_with_bbox)
            draw.rectangle(target_bbox, outline='red', width=3)
            draw2 = ImageDraw.Draw(artifact_with_bbox)
            draw2.rectangle(target_bbox, outline='green', width=3)

            result = artifact_explanation(
                client=client,
                real_image=original_with_bbox,
                artifact_image=artifact_with_bbox,
                entity=entity,
                part=subentity,
                artifact_type=artifact_type,
                money_manager=money_manager
            )

            if result['success']:
                artifact['explanation'] = result['explanation']
                artifacts_processed += 1
                logging.info(f"  ✓ Generated explanation for {entity}/{subentity}: {result['explanation'][:100]}...")
            else:
                artifact['explanation'] = ""
                artifacts_failed += 1
                logging.warning(f"  ✗ Failed to generate explanation for {entity}/{subentity}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            artifact['explanation'] = ""
            artifacts_failed += 1
            logging.error(f"  ✗ Error processing artifact {entity}/{subentity}: {str(e)}")

    # Save metadata once after processing all artifacts
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"  📁 {image_dir}: {artifacts_processed} explanations generated, {artifacts_failed} failed")
    return True, money_manager.total_cost, None
# -----------------------------
# Main Entry Point
# -----------------------------
def process_metadata_files():
    client = OpenAI()
    base_dir = "../exps/filtering/testing_multi_caption"

    image_dirs_all = []

    image_dirs = glob.glob(os.path.join(base_dir, "filtered_*"))
    image_dirs_all.extend([d for d in image_dirs])

    total_processed = 0
    total_errors = 0
    total_cost = 0.0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_single_image_dir, image_dir, client): image_dir
            for image_dir in image_dirs_all
        }

        for future in as_completed(futures):
            success, cost, error = future.result()
            if success:
                total_processed += 1
                total_cost += cost
            else:
                total_errors += 1
                logging.error(f"Error processing {futures[future]}: {error}")

    logging.info("=" * 50)
    logging.info("Processing complete!")
    logging.info(f"Total processed: {total_processed}")
    logging.info(f"Total errors: {total_errors}")
    logging.info(f"Total cost: ${total_cost:.4f}")
    logging.info("=" * 50)

if __name__ == "__main__":
    process_metadata_files()