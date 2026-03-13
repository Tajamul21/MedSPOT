import json
import math
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
from PIL import Image, ImageFile

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants

EARLY_STEP_ALPHA  = 0.8
PATCH_SIZE        = 14
QWEN_COORD_SCALE  = 1000.0

SMALL_TARGET_AREA  = 4e-4
MEDIUM_TARGET_AREA = 3e-3

EDGE_MARGIN          = 0.05
TOOLBAR_Y_THRESHOLD  = 0.12

# Geometry helpers

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
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def l2_norm_pixel(p, q, W, H):
    return math.sqrt(
        ((p[0] - q[0]) / W) ** 2 +
        ((p[1] - q[1]) / H) ** 2
    )


def resolve_image_path(dataset_root, image_path_from_json):
    return dataset_root / image_path_from_json


def sanity_check_images(tasks, dataset_root):
    missing = []
    for task in tasks:
        for step in task.get("steps", []):
            img_rel = step.get("image_path")
            if not img_rel:
                continue
            full = resolve_image_path(dataset_root, img_rel)
            if not full.exists():
                missing.append(str(full))
    if missing:
        print(f"Warning: {len(missing)} images missing. Example:\n{missing[0]}")


# Failure Taxonomy
def difficulty_tag(area):
    if area < SMALL_TARGET_AREA:
        return "small"
    if area < MEDIUM_TARGET_AREA:
        return "medium"
    return "large"


def classify_failure(pred_point, gt_bbox, W, H):
    if pred_point is None:
        return "no_prediction"

    area = bbox_area_ratio(gt_bbox, W, H)
    if area < SMALL_TARGET_AREA:
        return "small_target"

    px, py = pred_point
    cx, cy = bbox_center(gt_bbox)

    if px < EDGE_MARGIN * W or px > (1 - EDGE_MARGIN) * W:
        return "edge_bias"

    if py < TOOLBAR_Y_THRESHOLD * H:
        return "toolbar_confusion"

    if l2_norm_pixel((px, py), (cx, cy), W, H) < 0.03:
        return "near_miss"

    return "far_miss"


#Output Parsing
def extract_pixel_point(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    patterns = [
        r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)",
        r"x\s*=\s*([\d.]+)\s*,\s*y\s*=\s*([\d.]+)",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


# Step Evaluation
def evaluate_step(image, instruction, gt_bbox, model, processor):
    W, H = image.size

    prompt = (
        f"{instruction}\n"
        "Look at the image and click the correct UI element. "
        "Respond ONLY with the click location in the format (x,y)."
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt},
        ],
    }]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input], images=image_inputs, videos=video_inputs,
        return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=64, do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id)

    gen_ids  = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    raw_pred = extract_pixel_point(raw_text)

    if raw_pred is not None:
        px   = raw_pred[0] / QWEN_COORD_SCALE * W
        py   = raw_pred[1] / QWEN_COORD_SCALE * H
        pred = (px, py)
    else:
        pred = None

    x1, y1, x2, y2 = gt_bbox

    if pred is not None:
        px, py = pred
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True, "hit", pred, raw_text
        patch = [px - PATCH_SIZE / 2, py - PATCH_SIZE / 2,
                 px + PATCH_SIZE / 2, py + PATCH_SIZE / 2]
        if boxes_overlap(patch, gt_bbox):
            return True, "hit_overlap", pred, raw_text

    return False, classify_failure(pred, gt_bbox, W, H), pred, raw_text


# Task Evaluation

def evaluate_task(task, dataset_root, model, processor):
    steps     = task.get("steps", [])
    num_steps = len(steps)

    correct  = 0
    weighted = 0.0
    logs     = []

    for idx, step in enumerate(steps, start=1):
        img_path = resolve_image_path(dataset_root, step["image_path"])
        image    = Image.open(img_path).convert("RGB")

        gt_action = next(
            (a for a in step.get("actions", [])
             if a.get("type") == "click" and "bbox" in a), None)
        if gt_action is None:
            continue

        gt_bbox = percent_bbox_xywh_to_pixel_xyxy(gt_action["bbox"], *image.size)
        area    = bbox_area_ratio(gt_bbox, *image.size)

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

        correct  += 1
        weighted += EARLY_STEP_ALPHA ** (idx - 1)

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
    total         = len(results)
    completed     = sum(r["completed"]     for r in results)
    total_steps   = sum(r["num_steps"]     for r in results)
    correct_steps = sum(r["correct_steps"] for r in results)
    step1_hits    = sum(1 for r in results if r["steps"] and r["steps"][0]["hit"] == 1)

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

# MAIN
"""
Run:
    python evaluate_qwen3vl0.py \
        --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
        --data_dir           /path/to/annotations \
        --dataset_root       /path/to/dataset \
        --results_dir        /path/to/results
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--data_dir",     required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--results_dir",  required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    model_short  = Path(args.model_name_or_path).name
    RESULTS_ROOT = Path(args.results_dir) / model_short
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen3-VL model...")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype="auto", device_map="auto").eval()
    print("Model loaded.\n")

    all_results = []

    for json_file in sorted(Path(args.data_dir).rglob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        print(f"\nEvaluating {json_file.name}")
        print(f"Tasks: {len(data.get('tasks', []))}")

        sanity_check_images(data["tasks"], dataset_root)

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

        with open(run_dir / "task_results.json",  "w") as f: json.dump(results,  f, indent=2)
        with open(run_dir / "task_failures.json", "w") as f: json.dump(failures, f, indent=2)

        metrics = compute_metrics(results)
        with open(run_dir / "task_metrics.json", "w") as f: json.dump(metrics, f, indent=2)

        stats = failure_statistics(results)
        with open(run_dir / "failure_statistics.json", "w") as f: json.dump(stats, f, indent=2)

        print("Metrics:", metrics)

    overall_metrics = compute_metrics(all_results)
    with open(RESULTS_ROOT / "overall_dataset_metrics.json", "w") as f:
        json.dump(overall_metrics, f, indent=2)

    print("\nOVERALL DATASET METRICS")
    print(json.dumps(overall_metrics, indent=2))