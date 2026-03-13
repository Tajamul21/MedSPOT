import json
import math
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
from PIL import Image, ImageFile

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Prevent crashes on truncated/corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants


EARLY_STEP_ALPHA = 0.8
PATCH_SIZE = 14

SMALL_TARGET_AREA = 4e-4
MEDIUM_TARGET_AREA = 3e-3

NEAR_MISS_EXPANSION = 1.5
EDGE_MARGIN = 0.05
TOOLBAR_Y_THRESHOLD = 0.12


# Geometry helpers
def percent_bbox_xywh_to_pixel_xyxy(b, W, H):
    """Convert percent-format [x, y, w, h] (0–100) to pixel [x1, y1, x2, y2]."""
    x, y, w, h = b
    return [
        x / 100.0 * W,
        y / 100.0 * H,
        (x + w) / 100.0 * W,
        (y + h) / 100.0 * H,
    ]


def bbox_area_pixel(b, W, H):
    return ((b[2] - b[0]) / W) * ((b[3] - b[1]) / H)


def bbox_center_pixel(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def expand_bbox_pixel(b, factor, W, H):
    cx, cy = bbox_center_pixel(b)
    w = (b[2] - b[0]) * factor
    h = (b[3] - b[1]) * factor
    return [
        max(0.0, cx - w / 2.0),
        max(0.0, cy - h / 2.0),
        min(W, cx + w / 2.0),
        min(H, cy + h / 2.0),
    ]


def boxes_overlap(a, b):
    return not (
        a[2] < b[0] or
        a[0] > b[2] or
        a[3] < b[1] or
        a[1] > b[3]
    )


def l2(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


# Image resolution
def resolve_image_path(dataset_root, image_path_from_json):
    return dataset_root / image_path_from_json


def sanity_check_images(tasks, dataset_root):
    # Warn on missing images instead of raising RuntimeError so a single
    # bad file doesn't abort the entire evaluation run.
    missing = []
    for task in tasks:
        for step in task.get("steps", []):
            p = resolve_image_path(dataset_root, step["image_path"])
            if not p.exists():
                missing.append(str(p))
    if missing:
        print(f"Warning: {len(missing)} images missing. Example:\n{missing[0]}")



# Failure taxonomy


def classify_failure(pred_point, gt_bbox, W, H):
    if pred_point is None:
        return "no_prediction"

    area = bbox_area_pixel(gt_bbox, W, H)
    if area < SMALL_TARGET_AREA:
        return "small_target"

    expanded = expand_bbox_pixel(gt_bbox, NEAR_MISS_EXPANSION, W, H)
    px, py = pred_point

    if expanded[0] <= px <= expanded[2] and expanded[1] <= py <= expanded[3]:
        return "near_miss"

    if px < EDGE_MARGIN * W or px > (1 - EDGE_MARGIN) * W:
        return "edge_bias"

    if py < TOOLBAR_Y_THRESHOLD * H:
        return "toolbar_confusion"

    return "far_miss"


def difficulty_tag(area):
    if area < SMALL_TARGET_AREA:
        return "small"
    elif area < MEDIUM_TARGET_AREA:
        return "medium"
    return "large"



# Output parsing
def extract_click_point(text):
    """
    Parse (x, y) from model output.
    Strip <think>...</think> blocks first so we always parse the
    final answer, not a coordinate from inside the reasoning chain.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    patterns = [
        r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)",
        r"x\s*=\s*([\d.]+)\s*,\s*y\s*=\s*([\d.]+)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return [float(m.group(1)), float(m.group(2))]
    return None


# Qwen2.5-VL prediction

def predict_qwen_point(model, processor, image, instruction):
    """
    Run inference and return the predicted click point in pixel space.

    Uses the correct three-step Qwen-VL processor pipeline:
      1. apply_chat_template(tokenize=False) → text string with image tokens
      2. process_vision_info(messages)       → encoded image tensors
      3. processor(text=..., images=...)     → final model inputs

    Qwen2.5-VL outputs raw pixel coordinates directly — no rescaling needed.
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": instruction + "\nRespond ONLY with (x,y) pixel coordinates.",
            },
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
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    # Qwen2.5-VL outputs pixel coordinates directly 
    pred_point = extract_click_point(raw_text)

    return pred_point, raw_text



# Task evaluation
def evaluate_task(task, model, processor, dataset_root):
    steps = task["steps"]
    correct = 0
    weighted = 0.0
    logs = []

    for idx, step in enumerate(steps, start=1):
        image = Image.open(
            resolve_image_path(dataset_root, step["image_path"])
        ).convert("RGB")
        W, H = image.size

        gt_action = None
        for a in step.get("actions", []):
            if a.get("type") == "click" and "bbox" in a:
                gt_action = a
                break

        # Skip steps with no click action without aborting the whole task
        if gt_action is None:
            continue

        gt_bbox = percent_bbox_xywh_to_pixel_xyxy(gt_action["bbox"], W, H)
        area = bbox_area_pixel(gt_bbox, W, H)

        pred_point, raw = predict_qwen_point(
            model, processor, image, step["instruction"]
        )

        # Primary hit: predicted point lies inside gt_bbox
        hit = 0
        if pred_point is not None:
            x1, y1, x2, y2 = gt_bbox
            if x1 <= pred_point[0] <= x2 and y1 <= pred_point[1] <= y2:
                hit = 1

        # Secondary leniency: 14×14 patch around prediction overlaps gt_bbox
        if not hit and pred_point is not None:
            patch = [
                pred_point[0] - PATCH_SIZE / 2,
                pred_point[1] - PATCH_SIZE / 2,
                pred_point[0] + PATCH_SIZE / 2,
                pred_point[1] + PATCH_SIZE / 2,
            ]
            if boxes_overlap(patch, gt_bbox):
                hit = 1

        failure = None if hit else classify_failure(pred_point, gt_bbox, W, H)

        logs.append({
            "step_id": idx,
            "hit": hit,
            "failure_type": failure,
            "difficulty": difficulty_tag(area),
            "pred_point": pred_point,
            "raw_output": raw,
            "image_path": step["image_path"],
        })

        if not hit:
            break

        correct += 1
        weighted += EARLY_STEP_ALPHA ** (idx - 1)

    completed = int(correct == len(steps))

    return {
        "task_overview": task["task_overview"],
        "completed": completed,
        "correct_steps": correct,
        "num_steps": len(steps),
        "weighted_prefix_score": weighted,
        "steps": logs,
    }



# Metrics
def compute_metrics(results):
    total = len(results)
    completed = sum(r["completed"] for r in results)
    total_steps = sum(r["num_steps"] for r in results)
    correct_steps = sum(r["correct_steps"] for r in results)

    step1_hits = sum(
        1 for r in results if r["steps"] and r["steps"][0]["hit"] == 1
    )

    return {
        "task_completion_accuracy": completed / total if total else 0.0,
        "step1_accuracy": step1_hits / total if total else 0.0,
        "avg_correct_prefix": correct_steps / total if total else 0.0,
        "step_hit_rate": correct_steps / total_steps if total_steps else 0.0,
        "avg_weighted_prefix_score": (
            sum(r["weighted_prefix_score"] for r in results) / total if total else 0.0
        ),
    }


def failure_statistics(results):
    counter = Counter()
    by_difficulty = defaultdict(Counter)

    for r in results:
        if not r["completed"] and r["steps"]:
            step = r["steps"][-1]
            counter[step["failure_type"]] += 1
            by_difficulty[step["difficulty"]][step["failure_type"]] += 1

    return {
        "failure_type_histogram": dict(counter),
        "failure_by_difficulty": {k: dict(v) for k, v in by_difficulty.items()},
    }

# Main

"""
Run:
    python evaluate_qwen2_5vl.py \
        --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
        --data_dir /path/to/annotation/folder \
        --dataset_root /path/to/datasetroot
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing annotation JSON files")
    parser.add_argument("--dataset_root", required=True,
                        help="Root directory for resolving image paths in JSONs")
    parser.add_argument("--results_root", default="Results/Qwen25VL",
                        help="Directory where result JSONs will be written")

    args = parser.parse_args()
    args.dataset_root = Path(args.dataset_root)

    RESULTS_ROOT = Path(args.results_root)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
    ).eval()

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    all_results = []

    for json_file in sorted(Path(args.data_dir).glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        print(f"\nEvaluating {json_file.name} — {len(data.get('tasks', []))} tasks")

        sanity_check_images(data["tasks"], args.dataset_root)

        run_dir = RESULTS_ROOT / json_file.stem
        run_dir.mkdir(parents=True, exist_ok=True)

        results = []
        failures = []

        for task in tqdm(data["tasks"], desc=json_file.stem):
            r = evaluate_task(task, model, processor, args.dataset_root)
            results.append(r)
            if not r["completed"]:
                failures.append(r)

        all_results.extend(results)

        with open(run_dir / "task_results.json", "w") as f:
            json.dump(results, f, indent=2)

        with open(run_dir / "task_failures.json", "w") as f:
            json.dump(failures, f, indent=2)

        metrics = compute_metrics(results)
        with open(run_dir / "task_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        stats = failure_statistics(results)
        with open(run_dir / "failure_statistics.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"Finished {json_file.name}")
        print(json.dumps(metrics, indent=2))

    overall_metrics = compute_metrics(all_results)
    with open(RESULTS_ROOT / "overall_dataset_metrics.json", "w") as f:
        json.dump(overall_metrics, f, indent=2)

    print("\nOVERALL DATASET METRICS")
    print(json.dumps(overall_metrics, indent=2))