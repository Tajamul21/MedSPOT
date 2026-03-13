import json
import math
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
from PIL import Image, ImageFile

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants

early_step_alpha = 0.8
patch_size = 14

small_target_area = 4e-4
medium_target_area = 3e-3

edge_margin = 0.05
toolbar_y_threshold = 0.12


# Geometry

def percent_bbox_xywh_to_pixel_xyxy(b, W, H):
    x, y, w, h = b
    return [
        x / 100.0 * W,
        y / 100.0 * H,
        (x + w) / 100.0 * W,
        (y + h) / 100.0 * H,
    ]


def bbox_area_ratio(b, W, H):
    return ((b[2] - b[0]) / W) * ((b[3] - b[1]) / H)


def bbox_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def boxes_overlap(a, b):
    return not (
        a[2] < b[0] or
        a[0] > b[2] or
        a[3] < b[1] or
        a[1] > b[3]
    )


def l2_norm_pixel(p, q, W, H):
    return math.sqrt(
        ((p[0] - q[0]) / W) ** 2 +
        ((p[1] - q[1]) / H) ** 2
    )

# Failure taxonomy

def difficulty_tag(area):
    if area < small_target_area:
        return "small"
    if area < medium_target_area:
        return "medium"
    return "large"


def classify_failure(pred_point, gt_bbox, W, H):
    if pred_point is None:
        return "no_prediction"

    area = bbox_area_ratio(gt_bbox, W, H)
    if area < small_target_area:
        return "small_target"

    px, py = pred_point
    cx, cy = bbox_center(gt_bbox)

    if px < edge_margin * W or px > (1 - edge_margin) * W:
        return "edge_bias"

    if py < toolbar_y_threshold * H:
        return "toolbar_confusion"

    if l2_norm_pixel((px, py), (cx, cy), W, H) < 0.03:
        return "near_miss"

    return "far_miss"

# Output parsing


def extract_pixel_point(text):
    text = re.sub(r"<.*?>", "", text, flags=re.DOTALL)

    patterns = [
        r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)",
        r"x\s*=\s*([\d.]+)\s*,\s*y\s*=\s*([\d.]+)",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            return float(m.group(1)), float(m.group(2))

    return None

# Step evaluation

def evaluate_step(image, instruction, gt_bbox, model, processor):
    W, H = image.size

    prompt = (
        f"{instruction}\n"
        "Click anywhere inside the correct UI element's bounding box. "
        "The click does NOT need to be centered. "
        "Respond ONLY with the click location in format (x,y)."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=text_input, images=image, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=64, do_sample=False)

    raw_text = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0]

    pred = extract_pixel_point(raw_text)

    x1, y1, x2, y2 = gt_bbox

    if pred is not None:
        px, py = pred
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True, "hit", pred, raw_text
        patch = [px - patch_size / 2, py - patch_size / 2,
                 px + patch_size / 2, py + patch_size / 2]
        if boxes_overlap(patch, gt_bbox):
            return True, "hit_overlap", pred, raw_text

    return False, classify_failure(pred, gt_bbox, W, H), pred, raw_text



# Task evaluation

def evaluate_task(task, dataset_root, model, processor):
    steps = task.get("steps", [])
    num_steps = len(steps)

    correct = 0
    weighted = 0.0
    logs = []

    for idx, step in enumerate(steps, start=1):
        image = Image.open(dataset_root / step["image_path"]).convert("RGB")

        gt_action = next(
            (a for a in step.get("actions", [])
             if a.get("type") == "click" and "bbox" in a),
            None,
        )
        if gt_action is None:
            continue

        gt_bbox = percent_bbox_xywh_to_pixel_xyxy(gt_action["bbox"], *image.size)
        area = bbox_area_ratio(gt_bbox, *image.size)

        hit, reason, pred, raw = evaluate_step(
            image, step["instruction"], gt_bbox, model, processor)

        logs.append({
            "step_id":      idx,
            "hit":          int(hit),
            "failure_type": None if hit else reason,
            "difficulty":   difficulty_tag(area),
            "pred_point":   pred,
            "raw_output":   raw,
            "image_path":   step["image_path"],
        })

        if not hit:
            break

        correct += 1
        weighted += early_step_alpha ** (idx - 1)

    return {
        "task_overview":         task.get("task_overview", ""),
        "completed":             int(correct == num_steps),
        "correct_steps":         correct,
        "num_steps":             num_steps,
        "weighted_prefix_score": weighted,
        "steps":                 logs,
    }

# Metrics

def compute_metrics(results):
    total = len(results)
    completed     = sum(r["completed"]     for r in results)
    total_steps   = sum(r["num_steps"]     for r in results)
    correct_steps = sum(r["correct_steps"] for r in results)

    step1_hits = sum(
        1 for r in results if r["steps"] and r["steps"][0]["hit"] == 1
    )

    return {
        "task_completion_accuracy":  completed / total if total else 0.0,
        "step1_accuracy":            step1_hits / total if total else 0.0,
        "avg_correct_prefix":        correct_steps / total if total else 0.0,
        "step_hit_rate":             correct_steps / total_steps if total_steps else 0.0,
        "avg_weighted_prefix_score": sum(r["weighted_prefix_score"] for r in results) / total if total else 0.0,
    }


def failure_statistics(results):
    counter       = Counter()
    by_difficulty = defaultdict(Counter)

    for r in results:
        if not r["completed"] and r["steps"]:
            s = r["steps"][-1]
            counter[s["failure_type"]] += 1
            by_difficulty[s["difficulty"]][s["failure_type"]] += 1

    return {
        "failure_type_histogram": dict(counter),
        "failure_by_difficulty":  {k: dict(v) for k, v in by_difficulty.items()},
    }

# Main

"""
Run:
    python evaluate_gemma3_27B.py \
        --model_name_or_path google/gemma-3-27b-it \
        --data_dir           /path/to/annotations \
        --dataset_root       /path/to/dataset \
        --results_dir        /path/to/results
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="google/gemma-3-27b-it")
    parser.add_argument("--data_dir",     required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--results_dir",  required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    model_short  = Path(args.model_name_or_path).name
    RESULTS_ROOT = Path(args.results_dir) / model_short
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading Gemma 3 model...")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, use_fast=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto").eval()
    print("Model loaded.\n")

    all_results = []

    for json_file in sorted(Path(args.data_dir).rglob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        print(f"\nEvaluating {json_file.name}")

        run_dir = RESULTS_ROOT / json_file.stem
        run_dir.mkdir(parents=True, exist_ok=True)

        results  = []
        failures = []

        for task in tqdm(data["tasks"], desc=json_file.stem):
            r = evaluate_task(task, dataset_root, model, processor)
            results.append(r)
            if not r["completed"]:
                failures.append(r)

        all_results.extend(results)

        json.dump(results,                   open(run_dir / "task_results.json",      "w"), indent=2)
        json.dump(failures,                  open(run_dir / "task_failures.json",     "w"), indent=2)
        json.dump(compute_metrics(results),  open(run_dir / "task_metrics.json",      "w"), indent=2)
        json.dump(failure_statistics(results), open(run_dir / "failure_statistics.json", "w"), indent=2)

    overall_metrics = compute_metrics(all_results)
    json.dump(overall_metrics, open(RESULTS_ROOT / "overall_dataset_metrics.json", "w"), indent=2)

    print("\nOverall Dataset Metrics")
    print(json.dumps(overall_metrics, indent=2))