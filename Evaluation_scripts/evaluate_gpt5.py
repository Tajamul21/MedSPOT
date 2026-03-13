import json
import math
import re
import base64
import time
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
from PIL import Image, ImageFile
from io import BytesIO

from openai import OpenAI

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants

MODEL_NAME = "gpt-5"

DEBUG = True
DEBUG_MAX_PRINTS = 5

client = OpenAI()

EARLY_STEP_ALPHA = 0.8
PATCH_SIZE = 14

SMALL_TARGET_AREA = 4e-4
MEDIUM_TARGET_AREA = 3e-3

EDGE_MARGIN = 0.05
TOOLBAR_Y_THRESHOLD = 0.12



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
        a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3]
    )


def l2_norm_pixel(p, q, W, H):
    return math.sqrt(
        ((p[0] - q[0]) / W) ** 2 +
        ((p[1] - q[1]) / H) ** 2
    )



# Failure taxonomy


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


# Sanity Check
def sanity_check_images(tasks, dataset_root):
    missing = []
    for task in tasks:
        for step in task.get("steps", []):
            img_rel = step.get("image_path")
            if not img_rel:
                continue
            full = dataset_root / img_rel
            if not full.exists():
                missing.append(str(full))
    if missing:
        print(f"  [WARN] {len(missing)} missing image(s). First: {missing[0]}")


# Parsing
def extract_pixel_point(text):
    patterns = [
        r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)",
        r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]",
        r"x\s*=\s*([\d.]+)\s*[,;]?\s*y\s*=\s*([\d.]+)",
        r"x\s*:\s*([\d.]+)\s*[,;]?\s*y\s*:\s*([\d.]+)",
        r"(?:at|click|point|coord(?:inate)?s?)\s*[:\-]?\s*([\d.]+)\s*[,\s]\s*([\d.]+)",
        r"(?<!\d)([\d]{1,5})\s*[,\s]\s*([\d]{1,5})(?!\d)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


# Image Encoding
def encode_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# GPT Call
def call_gpt(image_b64, prompt, retries=3):
    last_error = None
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_b64}"
                        }
                    ],
                }],
            )

            if hasattr(response, "output_text") and response.output_text:
                return response.output_text

            text = ""
            for item in response.output:
                if hasattr(item, "type") and item.type == "output_text":
                    text += item.text
            if text:
                return text

            print(f"  [WARN] call_gpt: empty extraction. Full response: {response}")
            return ""

        except Exception as e:
            last_error = e
            print(f"  [ERROR] call_gpt attempt {attempt + 1}/{retries}: {type(e).__name__}: {e}")
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))

    print(f"  [ERROR] call_gpt: all {retries} attempts failed. Last error: {last_error}")
    return ""


# Step Evaluation
def normalize_prediction(pred, W, H):
    if pred is None:
        return None
    x, y = pred
    if 0 <= x <= W and 0 <= y <= H:
        return (x, y)
    return None


def evaluate_step(image, instruction, gt_bbox, debug_print=False):
    W, H = image.size
    img_b64 = encode_image(image)

    prompt = (
        f"{instruction}\n"
        f"Image resolution: width={W}, height={H}.\n"
        f"Click the correct UI element.\n"
        f"Return ONLY integer pixel coordinates as (x,y) "
        f"where x is in [0,{W}] and y is in [0,{H}]. No explanation."
    )

    raw_text = call_gpt(img_b64, prompt)

    if debug_print:
        print(f"    [DEBUG] raw model output : {repr(raw_text)}")

    pred = extract_pixel_point(raw_text)
    pred = normalize_prediction(pred, W, H)

    if debug_print:
        print(f"    [DEBUG] parsed prediction: {pred}  |  gt_bbox: {[round(v, 1) for v in gt_bbox]}")

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
def evaluate_task(task, dataset_root, debug_counter=None):
    steps = task.get("steps", [])
    correct = 0
    weighted = 0.0
    logs = []
    num_evaluable_steps = 0

    for idx, step in enumerate(steps, start=1):
        image = Image.open(dataset_root / step["image_path"]).convert("RGB")

        gt_action = next(
            (a for a in step.get("actions", []) if a.get("type") == "click" and "bbox" in a),
            None,
        )
        if gt_action is None:
            continue

        num_evaluable_steps += 1
        gt_bbox = percent_bbox_xywh_to_pixel_xyxy(gt_action["bbox"], *image.size)
        area = bbox_area_ratio(gt_bbox, *image.size)

        should_debug = (DEBUG and debug_counter is not None
                        and debug_counter[0] < DEBUG_MAX_PRINTS)

        hit, reason, pred, raw = evaluate_step(
            image, step["instruction"], gt_bbox, debug_print=should_debug)

        if should_debug and not hit:
            debug_counter[0] += 1

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
        weighted += EARLY_STEP_ALPHA ** (idx - 1)

    completed = int(correct == num_evaluable_steps) if num_evaluable_steps > 0 else 0
    step1_hit = int(len(logs) > 0 and logs[0]["hit"] == 1)

    return {
        "task_overview":         task.get("task_overview", ""),
        "completed":             completed,
        "step1_hit":             step1_hit,
        "correct_steps":         correct,
        "num_steps":             num_evaluable_steps,
        "weighted_prefix_score": weighted,
        "steps":                 logs,
    }


# Metrics
def compute_metrics(results):
    total = len(results)
    if total == 0:
        return {
            "task_completion_accuracy":  0.0,
            "step1_accuracy":            0.0,
            "avg_correct_prefix":        0.0,
            "step_hit_rate":             0.0,
            "avg_weighted_prefix_score": 0.0,
        }

    completed      = sum(r["completed"]            for r in results)
    step1_hits     = sum(r["step1_hit"]             for r in results)
    total_steps    = sum(r["num_steps"]             for r in results)
    correct_steps  = sum(r["correct_steps"]         for r in results)
    total_weighted = sum(r["weighted_prefix_score"] for r in results)

    return {
        "task_completion_accuracy":  completed     / total,
        "step1_accuracy":            step1_hits    / total,
        "avg_correct_prefix":        correct_steps / total,
        "step_hit_rate":             correct_steps / total_steps if total_steps else 0.0,
        "avg_weighted_prefix_score": total_weighted / total,
    }


def failure_statistics(results):
    counter       = Counter()
    by_difficulty = defaultdict(Counter)

    for r in results:
        if not r["completed"] and r["steps"]:
            s = r["steps"][-1]
            ft = s["failure_type"] or "unknown"
            diff = s.get("difficulty", "unknown")
            counter[ft] += 1
            by_difficulty[diff][ft] += 1

    return {
        "total_failed_tasks":     sum(counter.values()),
        "failure_type_histogram": dict(counter),
        "failure_by_difficulty":  {k: dict(v) for k, v in sorted(by_difficulty.items())},
    }


# Main
"""
Run:
    python evaluate_gpt5.py \
        --data_dir     /path/to/annotations \
        --dataset_root /path/to/dataset \
        --results_dir  /path/to/results
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     required=True,
                        help="Folder containing annotation JSON files")
    parser.add_argument("--dataset_root", required=True,
                        help="Root folder of the dataset images")
    parser.add_argument("--results_dir",  required=True,
                        help="Folder to save result JSONs")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    RESULTS_ROOT = Path(args.results_dir) / "GPT5"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    all_results  = []
    all_failures = []

    for json_file in sorted(Path(args.data_dir).rglob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        print(f"\nEvaluating {json_file.name}")
        print(f"Tasks: {len(data.get('tasks', []))}")

        sanity_check_images(data["tasks"], dataset_root)

        run_dir = RESULTS_ROOT / json_file.stem
        run_dir.mkdir(parents=True, exist_ok=True)

        results       = []
        failures      = []
        debug_counter = [0]

        for task in tqdm(data["tasks"], desc=json_file.stem):
            r = evaluate_task(task, dataset_root, debug_counter=debug_counter)
            results.append(r)
            if not r["completed"]:
                failures.append(r)

        all_results.extend(results)
        all_failures.extend(failures)

        with open(run_dir / "task_results.json",  "w") as f: json.dump(results,  f, indent=2)
        with open(run_dir / "task_failures.json", "w") as f: json.dump(failures, f, indent=2)

        metrics = compute_metrics(results)
        with open(run_dir / "task_metrics.json", "w") as f: json.dump(metrics, f, indent=2)

        stats = failure_statistics(results)
        with open(run_dir / "failure_statistics.json", "w") as f: json.dump(stats, f, indent=2)

        print("Metrics:", json.dumps(metrics, indent=2))
        print("Failure stats:", json.dumps(stats, indent=2))

    overall_metrics = compute_metrics(all_results)
    with open(RESULTS_ROOT / "overall_dataset_metrics.json", "w") as f:
        json.dump(overall_metrics, f, indent=2)

    overall_failure_stats = failure_statistics(all_results)
    with open(RESULTS_ROOT / "overall_failure_statistics.json", "w") as f:
        json.dump(overall_failure_stats, f, indent=2)

    with open(RESULTS_ROOT / "overall_task_failures.json", "w") as f:
        json.dump(all_failures, f, indent=2)

    print("\n OVERALL DATASET METRICS ")
    print(json.dumps(overall_metrics, indent=2))
    print("\n OVERALL FAILURE STATISTICS ")
    print(json.dumps(overall_failure_stats, indent=2))