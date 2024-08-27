import yaml
import logging
import time
from helpers import process_images, convert_to_coco
from calculate_metrics import evaluate_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Load configuration
    logging.info("Loading configuration...")
    start_time = time.time()
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # # Step 1: Convert ground truth images to JSON
    # logging.info("Processing ground truth images...")
    # start_step = time.time()
    # process_images(config, ground_truth=True)
    # logging.info(f"Ground truth images processed in {time.time() - start_step:.2f} seconds.")

    # Step 2: Convert prediction images to JSON
    logging.info("Processing prediction images...")
    start_step = time.time()
    process_images(config, ground_truth=False)
    logging.info(f"Prediction images processed in {time.time() - start_step:.2f} seconds.")

    # # Step 3: Convert ground truth JSON to COCO format
    # logging.info("Converting ground truth JSON to COCO format...")
    # start_step = time.time()
    # convert_to_coco(config, ground_truth=True)
    # logging.info(f"Ground truth JSON converted to COCO format in {time.time() - start_step:.2f} seconds.")

    # Step 4: Convert prediction JSON to COCO format
    logging.info("Converting prediction JSON to COCO format...")
    start_step = time.time()
    convert_to_coco(config, ground_truth=False)
    logging.info(f"Prediction JSON converted to COCO format in {time.time() - start_step:.2f} seconds.")

    # Step 5: Evaluate bbox metrics
    logging.info("Evaluating bbox metrics...")
    start_step = time.time()
    metrics = evaluate_metrics(config)
    #iou = evaluator.evaluate()
    logging.info(f"bbox metrics evaluated in {time.time() - start_step:.2f} seconds.")

    # Output the results
    print(metrics)

    # Total execution time
    logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()