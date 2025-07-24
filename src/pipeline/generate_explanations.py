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
def process_single_image_dir(image_dir, client, base_dir):
    metadata_path = os.path.join(image_dir, "metadata.json")
    original_image_path = os.path.join(image_dir, "original_image.png")
    artifact_image_path = os.path.join(image_dir, "artifact_with_bbox.png")

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        logging.info(f"Processing {os.path.basename(image_dir)}")

        original_image = Image.open(original_image_path)
        artifact_image = Image.open(artifact_image_path)
        draw = ImageDraw.Draw(original_image)
        draw.rectangle(metadata['target_bbox'], outline='red', width=3)

        money_manager = MoneyManager(model="gpt-4o")

        result = artifact_explanation(
            client=client,
            real_image=original_image,
            artifact_image=artifact_image,
            metadata=metadata,
            money_manager=money_manager
        )

        if result['success']:
            metadata['explanation'] = result['explanation']
            logging.info(f"  ✓ Generated explanation: {result['explanation'][:100]}...")
        else:
            metadata['explanation'] = ""
            logging.warning(f"  ✗ Failed to generate explanation: {result.get('error', 'Unknown error')}")

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return True, money_manager.total_cost, None

    except Exception as e:
        return False, 0.0, str(e)

# -----------------------------
# Main Entry Point
# -----------------------------
def process_metadata_files():
    client = OpenAI()
    base_dir = "../../exps/filtering"

    image_dirs_all = []

    for artifact_type in ["addition", "removal", "distortion"]:
        artifact_dir = os.path.join(base_dir, artifact_type)
        if not os.path.exists(artifact_dir):
            logging.warning(f"Directory {artifact_dir} does not exist, skipping...")
            continue

        logging.info(f"Processing {artifact_type} artifacts...")
        image_dirs = glob.glob(os.path.join(artifact_dir, "filtered_image_*"))
        image_dirs_all.extend([d for d in image_dirs])

    total_processed = 0
    total_errors = 0
    total_cost = 0.0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_single_image_dir, image_dir, client, base_dir): image_dir
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