import os
import json
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
from PIL import Image

import torch
from transformers import AutoProcessor
from gui_actor.constants import DEFAULT_POINTER_PAD_TOKEN, DEFAULT_POINTER_END_TOKEN
from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.inference import inference, ForceFollowTokensLogitsProcessor
from gui_actor.utils import do_boxes_overlap

# Constants
IMAGE_PATCH_SIZE = 14
EARLY_STEP_ALPHA = 0.8

SMALL_TARGET_AREA = 4e-4
MEDIUM_TARGET_AREA = 3e-3

NEAR_MISS_EXPANSION = 1.5
EDGE_MARGIN = 0.05
TOOLBAR_Y_THRESHOLD = 0.12

# Geometry helpers
def percent_bbox_xywh_to_norm_xyxy(b):
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

# Image path resolution
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

# Model loading
def load_model(model_path):
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()

    logits_processor = ForceFollowTokensLogitsProcessor(
        token_a_id=tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0],
        forced_sequence=[tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]],
    )

    prompt = (
        "You are a GUI agent. Locate the UI element described by the instruction "
        "and click inside it."
    )

    return model, tokenizer, processor, logits_processor, prompt

# Failure taxonomy
def classify_failure(pred_points, gt_bbox):
    if not pred_points:
        return "no_prediction"

    area = bbox_area(gt_bbox)
    if area < SMALL_TARGET_AREA:
        return "small_target"

    expanded = expand_bbox(gt_bbox, NEAR_MISS_EXPANSION)

    for px, py in pred_points:
        if expanded[0] <= px <= expanded[2] and expanded[1] <= py <= expanded[3]:
            return "near_miss"

    for px, py in pred_points:
        if px < EDGE_MARGIN or px > 1.0 - EDGE_MARGIN or py < EDGE_MARGIN or py > 1.0 - EDGE_MARGIN:
            return "edge_bias"

    for px, py in pred_points:
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
def evaluate_step(model, tokenizer, processor, logits_processor,
                  prompt, image, instruction, gt_bbox, args):

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction},
        ]},
    ]

    pred = inference(
        conversation,
        model,
        tokenizer,
        processor,
        logits_processor=logits_processor,
        use_placeholder=args.use_placeholder,
        topk=args.topk,
    )

    points = pred.get("topk_points", [])
    w, h = image.size
    x1, y1, x2, y2 = gt_bbox

    for px, py in points:
        px = min(max(px, 0.0), 1.0)
        py = min(max(py, 0.0), 1.0)

        if x1 <= px <= x2 and y1 <= py <= y2:
            return True, "hit", points

        patch = [
            px - IMAGE_PATCH_SIZE / w,
            py - IMAGE_PATCH_SIZE / h,
            px + IMAGE_PATCH_SIZE / w,
            py + IMAGE_PATCH_SIZE / h,
        ]

        if do_boxes_overlap(patch, gt_bbox):
            return True, "hit_overlap", points

    return False, classify_failure(points, gt_bbox), points

# Task evaluation
def evaluate_task(task, model, tokenizer, processor, logits_processor, prompt, args):
    steps = task.get("steps", [])
    num_steps = len(steps)

    correct = 0
    weighted = 0.0
    logs = []

    for idx, step in enumerate(steps, start=1):
        image = Image.open(
            resolve_image_path(args.dataset_root, step["image_path"])
        ).convert("RGB")

        instruction = step.get("instruction", "")

        gt_action = None
        for a in step.get("actions", []):
            if a.get("type") == "click" and "bbox" in a:
                gt_action = a
                break

        if gt_action is None:
            break

        gt_bbox = percent_bbox_xywh_to_norm_xyxy(gt_action["bbox"])
        area = bbox_area(gt_bbox)

        hit, reason, points = evaluate_step(
            model, tokenizer, processor, logits_processor,
            prompt, image, instruction, gt_bbox, args
        )

        logs.append({
            "step_id": idx,
            "hit": int(hit),
            "failure_type": None if hit else reason,
            "difficulty": difficulty_tag(area),
            "gt_bbox_area": area,
            "min_l2_distance": min(
                [l2(p, bbox_center(gt_bbox)) for p in points], default=None
            ),
            "pred_points": points,
            "image_path": step["image_path"],
        })

        if not hit:
            break

        correct += 1
        weighted += EARLY_STEP_ALPHA ** (idx - 1)

    completed = int(correct == num_steps)

    return {
        "task_overview": task.get("task_overview", ""),
        "completed": completed,
        "correct_steps": correct,
        "num_steps": num_steps,
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
        "avg_weighted_prefix_score": sum(
            r["weighted_prefix_score"] for r in results
        ) / total if total else 0.0,
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
        python evaluate_gui_actor.py \
        --model_name_or_path microsoft/GUI-Actor-7B-Qwen2.5-VL \
        --data_dir path/to/annotation/folder \
        --dataset_root path/to/dataset_root
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default="microsoft/GUI-Actor-7B-Qwen2.5-VL")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--use_placeholder", action="store_true", default=True)

    args = parser.parse_args()
    args.dataset_root = Path(args.dataset_root)

    REPO_ROOT = Path(__file__).resolve().parents[2]
    RESULTS_ROOT = REPO_ROOT / "Results" / "GUI-Actor"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    model, tokenizer, processor, logits_processor, prompt = load_model(
        args.model_name_or_path
    )

    all_results = []

    for json_file in sorted(Path(args.data_dir).glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        sanity_check_images(data["tasks"], args.dataset_root)

        run_dir = RESULTS_ROOT / json_file.stem
        run_dir.mkdir(parents=True, exist_ok=True)

        results = []
        failures = []

        for task in tqdm(data["tasks"], desc=json_file.stem):
            r = evaluate_task(
                task, model, tokenizer, processor, logits_processor, prompt, args
            )
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
        print(metrics)

    overall_metrics = compute_metrics(all_results)
    with open(RESULTS_ROOT / "overall_dataset_metrics.json", "w") as f:
        json.dump(overall_metrics, f, indent=2)

    print("\n===== OVERALL DATASET METRICS =====")
    print(json.dumps(overall_metrics, indent=2))
