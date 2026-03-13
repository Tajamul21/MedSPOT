import json
import math
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
from PIL import Image, ImageFile

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants 
IMAGE_PATCH_SIZE   = 14
EARLY_STEP_ALPHA   = 0.8

SMALL_TARGET_AREA  = 4e-4
MEDIUM_TARGET_AREA = 3e-3

NEAR_MISS_EXPANSION  = 1.5
EDGE_MARGIN          = 0.05
TOOLBAR_Y_THRESHOLD  = 0.12

# OS-Atlas coordinate range
OS_ATLAS_COORD_RANGE = 1000.0


# Geometry helpers 

def percent_bbox_xywh_to_norm_xyxy(b):
    """Convert [x, y, w, h] in percent → [x1, y1, x2, y2] normalized 0-1."""
    x, y, w, h = b
    return x / 100.0, y / 100.0, (x + w) / 100.0, (y + h) / 100.0


def bbox_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def bbox_center(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def expand_bbox(b, factor):
    cx, cy = bbox_center(b)
    w = (b[2] - b[0]) * factor
    h = (b[3] - b[1]) * factor
    return [
        max(0.0, cx - w / 2.0),
        max(0.0, cy - h / 2.0),
        min(1.0, cx + w / 2.0),
        min(1.0, cy + h / 2.0),
    ]


def l2(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def do_boxes_overlap(a, b):
    return not (
        a[2] < b[0] or a[0] > b[2] or
        a[3] < b[1] or a[1] > b[3]
    )


# Path helpers 

def resolve_image_path(dataset_root, image_path_from_json):
    return dataset_root / image_path_from_json


def sanity_check_images(tasks, dataset_root):
    missing = []
    for task in tasks:
        for step in task.get("steps", []):
            img = step.get("image_path")
            if not img:
                continue
            p = resolve_image_path(dataset_root, img)
            if not p.exists():
                missing.append(str(p))
    if missing:
        raise RuntimeError(f"{len(missing)} images missing. Example:\n{missing[0]}")


# Loading Model 

def load_model(model_path):
    print(f"Loading OS-Atlas model: {model_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    print("Model loaded.\n")
    return model, processor


def extract_point_from_text(text):
    """
    Parse OS-Atlas output and return normalized 0-1 (px, py).
    Coordinates are in 0-1000 range — divide by 1000.
    """
    # Strip think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # point format: [[x, y]]
    point_match = re.search(
        r"\[\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\]", text
    )
    if point_match:
        px = float(point_match.group(1)) / OS_ATLAS_COORD_RANGE
        py = float(point_match.group(2)) / OS_ATLAS_COORD_RANGE
        px = min(max(px, 0.0), 1.0)
        py = min(max(py, 0.0), 1.0)
        return px, py

    # bbox format: [[x1, y1, x2, y2]] — use center
    bbox_match = re.search(
        r"\[\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\]", text
    )
    if bbox_match:
        x1 = float(bbox_match.group(1)) / OS_ATLAS_COORD_RANGE
        y1 = float(bbox_match.group(2)) / OS_ATLAS_COORD_RANGE
        x2 = float(bbox_match.group(3)) / OS_ATLAS_COORD_RANGE
        y2 = float(bbox_match.group(4)) / OS_ATLAS_COORD_RANGE
        px = (x1 + x2) / 2.0
        py = (y1 + y2) / 2.0
        px = min(max(px, 0.0), 1.0)
        py = min(max(py, 0.0), 1.0)
        return px, py

    # Fallback: plain (x, y)
    fallback = re.search(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", text)
    if fallback:
        px = float(fallback.group(1)) / OS_ATLAS_COORD_RANGE
        py = float(fallback.group(2)) / OS_ATLAS_COORD_RANGE
        # if values already 0-1 range (not 0-1000), don't divide 
        if px > 1.0 or py > 1.0:
            pass  
        else:
            pass
        px = min(max(px, 0.0), 1.0)
        py = min(max(py, 0.0), 1.0)
        return px, py

    return None


# Failure taxonomy (all in normalized 0-1 space) 

def classify_failure(pred_point, gt_bbox):
    if pred_point is None:
        return "no_prediction"

    area = bbox_area(gt_bbox)
    if area < SMALL_TARGET_AREA:
        return "small_target"

    expanded = expand_bbox(gt_bbox, NEAR_MISS_EXPANSION)
    px, py = pred_point

    if expanded[0] <= px <= expanded[2] and expanded[1] <= py <= expanded[3]:
        return "near_miss"

    if px < EDGE_MARGIN or px > 1.0 - EDGE_MARGIN or py < EDGE_MARGIN or py > 1.0 - EDGE_MARGIN:
        return "edge_bias"

    if py < TOOLBAR_Y_THRESHOLD:
        return "toolbar_confusion"

    return "far_miss"


def difficulty_tag(area):
    if area < SMALL_TARGET_AREA:
        return "small"
    elif area < MEDIUM_TARGET_AREA:
        return "medium"
    return "large"


# Step evaluation 

def evaluate_step(model, processor, image_path, instruction, gt_bbox):
    """
    gt_bbox : normalized xyxy (0-1)
    returns : (hit: bool, reason: str, pred_point: tuple|None, W: int, H: int)
    """
    image = Image.open(image_path).convert("RGB")
    W, H  = image.size

    # OS-Atlas specific prompt format
    prompt = (
        f'In this UI screenshot, what is the position of the element '
        f'corresponding to the command "{instruction}" (with point)?'
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt},
        ],
    }]

    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    pred_text = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    print("Raw output:", pred_text)

    pred_point = extract_point_from_text(pred_text)
    print("Parsed normalized coordinate:", pred_point)

    if pred_point is not None:
        px, py = pred_point
        x1, y1, x2, y2 = gt_bbox

        # HIT
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True, "hit", pred_point, W, H

        # Patch overlap
        patch = [
            px - IMAGE_PATCH_SIZE / W,
            py - IMAGE_PATCH_SIZE / H,
            px + IMAGE_PATCH_SIZE / W,
            py + IMAGE_PATCH_SIZE / H,
        ]
        if do_boxes_overlap(patch, list(gt_bbox)):
            return True, "hit_overlap", pred_point, W, H

    return False, classify_failure(pred_point, gt_bbox), pred_point, W, H


# Task evaluation 

def evaluate_task(task, model, processor, dataset_root):
    steps           = task.get("steps", [])
    evaluable_steps = 0
    correct         = 0
    weighted        = 0.0
    logs            = []

    for idx, step in enumerate(steps, start=1):
        image_path  = resolve_image_path(dataset_root, step["image_path"])
        instruction = step.get("instruction", "")

        gt_action = next(
            (a for a in step.get("actions", [])
             if a.get("type") in ("click", "double_click", "drag") and "bbox" in a),
            None,
        )

        if gt_action is None:
            continue

        evaluable_steps += 1

        gt_bbox = percent_bbox_xywh_to_norm_xyxy(gt_action["bbox"])
        area    = bbox_area(gt_bbox)

        hit, reason, pred_point, W, H = evaluate_step(
            model, processor, image_path, instruction, gt_bbox
        )

        logs.append({
            "step_id":      idx,
            "action_type":  gt_action.get("type"),
            "hit":          int(hit),
            "failure_type": None if hit else reason,
            "difficulty":   difficulty_tag(area),
            "gt_bbox_area": area,
            "gt_bbox":      list(gt_bbox),
            "pred_point":   list(pred_point) if pred_point else None,
            "l2_distance":  l2(pred_point, bbox_center(gt_bbox)) if pred_point else None,
            "image_path":   step["image_path"],
            "image_w":      W,
            "image_h":      H,
        })

        if not hit:
            break

        correct  += 1
        weighted += EARLY_STEP_ALPHA ** (idx - 1)

    completed = int(correct == evaluable_steps)

    return {
        "task_overview":         task.get("task_overview", ""),
        "completed":             completed,
        "correct_steps":         correct,
        "num_steps":             evaluable_steps,
        "weighted_prefix_score": weighted,
        "steps":                 logs,
    }


# Metrics 

def compute_metrics(results):
    total = len(results)
    if total == 0:
        return {}

    completed     = sum(r["completed"]     for r in results)
    total_steps   = sum(r["num_steps"]     for r in results)
    correct_steps = sum(r["correct_steps"] for r in results)

    step1_hits = sum(
        1 for r in results if r["steps"] and r["steps"][0]["hit"] == 1
    )

    return {
        "task_completion_accuracy":  completed / total,
        "step1_accuracy":            step1_hits / total,
        "step_hit_rate":             correct_steps / total_steps if total_steps else 0.0,
        "avg_weighted_prefix_score": sum(
            r["weighted_prefix_score"] for r in results
        ) / total,
    }


def failure_statistics(results):
    counter       = Counter()
    by_difficulty = defaultdict(Counter)

    for r in results:
        if not r["completed"] and r["steps"]:
            step = r["steps"][-1]
            counter[step["failure_type"]] += 1
            by_difficulty[step["difficulty"]][step["failure_type"]] += 1

    return {
        "failure_type_histogram": dict(counter),
        "failure_by_difficulty":  {k: dict(v) for k, v in by_difficulty.items()},
    }


# Main 
"""
Run:
    python evaluate_osatlas.py \
        --model_name_or_path OS-Copilot/OS-Atlas-Base-7B \
        --data_dir /path/to/annotation/folder \
        --dataset_root /path/to/datasetroot
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="OS-Copilot/OS-Atlas-Base-7B")
    parser.add_argument("--data_dir",      required=True)
    parser.add_argument("--dataset_root",  required=True)
    args = parser.parse_args()

    args.dataset_root = Path(args.dataset_root)

    REPO_ROOT    = Path(__file__).resolve().parent
    RESULTS_ROOT = REPO_ROOT / "Results" / "OS-Atlas-7B"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.model_name_or_path)

    all_results = []

    for json_file in sorted(Path(args.data_dir).glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        sanity_check_images(data["tasks"], args.dataset_root)

        run_dir = RESULTS_ROOT / json_file.stem
        run_dir.mkdir(parents=True, exist_ok=True)

        results  = []
        failures = []

        for task in tqdm(data["tasks"], desc=json_file.stem):
            r = evaluate_task(task, model, processor, args.dataset_root)
            results.append(r)
            if not r["completed"]:
                failures.append(r)

        all_results.extend(results)

        with open(run_dir / "task_results.json",       "w") as f:
            json.dump(results, f, indent=2)
        with open(run_dir / "task_failures.json",      "w") as f:
            json.dump(failures, f, indent=2)

        metrics = compute_metrics(results)
        with open(run_dir / "task_metrics.json",       "w") as f:
            json.dump(metrics, f, indent=2)

        stats = failure_statistics(results)
        with open(run_dir / "failure_statistics.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\nFinished {json_file.name}")
        print(metrics)

    overall_metrics = compute_metrics(all_results)
    with open(RESULTS_ROOT / "overall_dataset_metrics.json", "w") as f:
        json.dump(overall_metrics, f, indent=2)

    print("\nOverall Dataset Metrics")
    print(json.dumps(overall_metrics, indent=2))