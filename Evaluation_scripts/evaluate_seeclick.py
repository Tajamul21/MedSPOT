import os
import json
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from process_utils import pred_2_point, extract_bbox

EARLY_STEP_ALPHA = 0.8

SMALL_TARGET_AREA = 4e-4
MEDIUM_TARGET_AREA = 3e-3

NEAR_MISS_EXPANSION = 1.5
EDGE_MARGIN = 0.05
TOOLBAR_Y_THRESHOLD = 0.12

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
        if px < EDGE_MARGIN or px > 1.0 - EDGE_MARGIN or \
           py < EDGE_MARGIN or py > 1.0 - EDGE_MARGIN:
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

def load_model(model_path, lora_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if lora_path:
        model = AutoModelForCausalLM.from_pretrained(
            lora_path,
            device_map="cuda",
            trust_remote_code=True,
            bf16=True
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            trust_remote_code=True,
            bf16=True
        ).eval()

    model.generation_config = GenerationConfig.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    prompt_template = (
        "In this UI screenshot, what is the position of the element "
        "corresponding to the command \"{}\" (with point)?"
    )

    return model, tokenizer, prompt_template

def evaluate_step(model, tokenizer, prompt_template,
                  image_path, instruction, gt_bbox):

    prompt = prompt_template.format(instruction)

    query = tokenizer.from_list_format([
        {'image': image_path},
        {'text': prompt}
    ])

    response, _ = model.chat(tokenizer, query=query, history=None)

    click_point = None

    try:
        if "box" in response:
            pred_bbox = extract_bbox(response)
            click_point = [
                (pred_bbox[0][0] + pred_bbox[1][0]) / 2,
                (pred_bbox[0][1] + pred_bbox[1][1]) / 2
            ]
            click_point = [p / 1000 for p in click_point]
        else:
            click_point = pred_2_point(response)
    except:
        return False, "no_prediction", []

    if click_point is None:
        return False, "no_prediction", []

    px = min(max(click_point[0], 0.0), 1.0)
    py = min(max(click_point[1], 0.0), 1.0)

    x1, y1, x2, y2 = gt_bbox

    if x1 <= px <= x2 and y1 <= py <= y2:
        return True, "hit", [(px, py)]

    failure = classify_failure([(px, py)], gt_bbox)

    return False, failure, [(px, py)]

def evaluate_task(task, model, tokenizer, prompt_template, args):

    steps = task.get("steps", [])
    num_steps = len(steps)

    correct = 0
    weighted = 0.0
    logs = []

    for idx, step in enumerate(steps, start=1):

        image_path = args.dataset_root / step["image_path"]
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
            model,
            tokenizer,
            prompt_template,
            str(image_path),
            instruction,
            gt_bbox
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

"""
Run :
    python evaluate_seeclick.py \
    --model_name_or_path path/to/seeclick/checkpoints \
    --lora_path path/to/lora/path \
    --data_dir path/to/annotation/folder \
    --dataset_root path/to/dataset_root
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--lora_path",required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dataset_root", required=True)

    args = parser.parse_args()
    args.dataset_root = Path(args.dataset_root)

    model, tokenizer, prompt_template = load_model(
        args.model_name_or_path,
        args.lora_path
    )

    all_results = []

    for json_file in sorted(Path(args.data_dir).glob("*.json")):

        with open(json_file) as f:
            data = json.load(f)

        results = []
        failures = []

        for task in tqdm(data["tasks"], desc=json_file.stem):

            r = evaluate_task(
                task,
                model,
                tokenizer,
                prompt_template,
                args
            )

            results.append(r)

            if not r["completed"]:
                failures.append(r)

        all_results.extend(results)

        out_dir = Path("Results") / "SeeClick" / json_file.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "task_results.json", "w") as f:
            json.dump(results, f, indent=2)

        with open(out_dir / "task_failures.json", "w") as f:
            json.dump(failures, f, indent=2)

        metrics = compute_metrics(results)
        with open(out_dir / "task_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        stats = failure_statistics(results)
        with open(out_dir / "failure_statistics.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\nFinished {json_file.name}")
        print(json.dumps(metrics, indent=2))

    overall_metrics = compute_metrics(all_results)

    Path("Results/SeeClick").mkdir(parents=True, exist_ok=True)
    with open("Results/SeeClick/overall_dataset_metrics.json", "w") as f:
        json.dump(overall_metrics, f, indent=2)

    print("\nOVERALL DATASET METRICS")
    print(json.dumps(overall_metrics, indent=2))
