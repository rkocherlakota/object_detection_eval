import json
import numpy as np
import yaml

def evaluate_metrics(config, iou_threshold=0.5):
    """
    Calculates IoU, Precision, Recall, and F1 Score for object detection from JSON files.

    Args:
    config (dict): Configuration dictionary containing paths to the JSON files.
    iou_threshold (float): IoU threshold to determine true positives.

    Returns:
    dict: Dictionary containing Average IoU, Precision, Recall, and F1 Score.
    """

    def calculate_iou_bbox(box1, box2):
        """Calculates IoU between two bounding boxes."""
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / union_area if union_area != 0 else 0
        return iou

    def load_bounding_boxes(json_file):
        """
        Loads bounding boxes from a JSON file and organizes them by image_id.

        Args:
        json_file (str): Path to the JSON file containing bounding boxes.

        Returns:
        dict: Dictionary of lists with image_id as keys and bounding boxes as values.
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        bboxes_by_image = {}
        for annotation in data['annotations']:
            image_id = annotation['image_id']
            bbox = annotation['bbox']
            x1, y1, width, height = bbox
            x2 = x1 + width
            y2 = y1 + height
            bbox = [x1, y1, x2, y2]
            if image_id not in bboxes_by_image:
                bboxes_by_image[image_id] = []
            bboxes_by_image[image_id].append(bbox)
        
        return bboxes_by_image

    # Extract file paths from config
    gt_json_file = config['input']['ground_truth_coco_output']
    pred_json_file = config['input']['prediction_coco_output']

    # Load bounding boxes from the JSON files
    gt_boxes_by_image = load_bounding_boxes(gt_json_file)
    pred_boxes_by_image = load_bounding_boxes(pred_json_file)

    matched_gt = []
    true_positive = 0
    total_iou = 0

    for image_id, pred_boxes in pred_boxes_by_image.items():
        gt_boxes = gt_boxes_by_image.get(image_id, [])
        for pred_box in pred_boxes:
            for gt_box in gt_boxes:
                iou = calculate_iou_bbox(pred_box, gt_box)
                total_iou += iou
                if iou >= iou_threshold and gt_box not in matched_gt:
                    true_positive += 1
                    matched_gt.append(gt_box)
                    break

    avg_iou = total_iou / len(pred_boxes_by_image) if pred_boxes_by_image else 0
    precision = true_positive / sum(len(v) for v in pred_boxes_by_image.values()) if pred_boxes_by_image else 0
    recall = true_positive / sum(len(v) for v in gt_boxes_by_image.values()) if gt_boxes_by_image else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return {
        "Average IoU": avg_iou,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
