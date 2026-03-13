import json
import math
import re
import argparse
import gc
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
from PIL import Image, ImageFile

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Constants

EARLY_STEP_ALPHA      = 0.8          
PATCH_FRAC            = 0.02        
MAX_RETRY_ATTEMPTS    = 2            
RETRY_TEMPERATURE     = 0.3

SMALL_TARGET_AREA     = 4e-4
MEDIUM_TARGET_AREA    = 3e-3

EDGE_MARGIN           = 0.05
TOOLBAR_Y_THRESHOLD   = 0.12

NORMALIZED_COORD_MAX  = 1.0

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
    return math.sqrt(((p[0] - q[0]) / W) ** 2 + ((p[1] - q[1]) / H) ** 2)


# Coordinate-scale detection
def maybe_denormalize(pred, W, H):
    """
    If both predicted coordinates are in (0, 1], treat them as normalised
    and convert to pixel space. This handles UI-TARS variants that output
    fractional coordinates rather than absolute pixel values.
    Returns (px, py, was_normalised).
    """
    px, py = pred
    if 0.0 <= px <= NORMALIZED_COORD_MAX and 0.0 <= py <= NORMALIZED_COORD_MAX:
    
        if px < 2.0 and py < 2.0:
            return px * W, py * H, True
    return px, py, False

# Image resolution

def resolve_image_path(dataset_root, image_path):
    return dataset_root / image_path


def sanity_check_images(tasks, dataset_root):
    """Collect ALL missing images before starting, not just the first."""
    missing = []
    for t in tasks:
        for s in t.get("steps", []):
            p = resolve_image_path(dataset_root, s["image_path"])
            if not p.exists():
                missing.append(str(p))
    if missing:
        msg = "\n".join(missing[:20])
        if len(missing) > 20:
            msg += f"\n... and {len(missing) - 20} more"
        raise RuntimeError(f"{len(missing)} images missing:\n{msg}")



# Output parsing

def extract_pixel_point(text):
    """
    Try several coordinate formats in order of specificity.
    Returns (x, y) floats or None.
    """
    patterns = [
        # CLICK(x, y)
        r"CLICK\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)",
        # [x, y] box format
        r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,[\d.\s,]+)?\]",
        # (x, y)
        r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)",
        # x=…, y=…
        r"x\s*=\s*([\d.]+)\s*,\s*y\s*=\s*([\d.]+)",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return float(m.group(1)), float(m.group(2))
    return None



# Failure taxonomy
def difficulty_tag(area):
    if area < SMALL_TARGET_AREA:
        return "small"
    if area < MEDIUM_TARGET_AREA:
        return "medium"
    return "large"


def classify_failure(pred_point, gt_bbox, W, H):
    """
    Extended taxonomy that distinguishes near/far misses on small targets.
    """
    if pred_point is None:
        return "no_prediction"

    area = bbox_area_ratio(gt_bbox, W, H)
    px, py = pred_point
    cx, cy = bbox_center(gt_bbox)
    dist   = l2_norm_pixel((px, py), (cx, cy), W, H)

    if area < SMALL_TARGET_AREA:
        return "small_target_near" if dist < 0.05 else "small_target_far"

    if px < EDGE_MARGIN * W or px > (1 - EDGE_MARGIN) * W:
        return "edge_bias"

    if py < TOOLBAR_Y_THRESHOLD * H:
        return "toolbar_confusion"

    if dist < 0.03:
        return "near_miss"

    return "far_miss"


# Loading Model
def load_model(model_path):
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    ).eval()
    return model, processor

# Step evaluation

def _run_inference(model, processor, image, prompt, do_sample=False, temperature=None):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=64,
        do_sample=do_sample,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    if do_sample and temperature is not None:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0]


def evaluate_step(model, processor, image, instruction, gt_bbox):
    W, H = image.size

    prompt = (
        f"{instruction}\n"
        "Look at the image and click the correct UI element. "
        "Respond ONLY with the click location in the format (x,y) "
        "using pixel coordinates."
    )

    # --- attempt 1: greedy ---
    text = _run_inference(model, processor, image, prompt, do_sample=False)
    raw_pred = extract_pixel_point(text)

    # --- retries with sampling if no coordinate found ---
    attempt = 0
    while raw_pred is None and attempt < MAX_RETRY_ATTEMPTS:
        attempt += 1
        text = _run_inference(
            model, processor, image, prompt,
            do_sample=True, temperature=RETRY_TEMPERATURE,
        )
        raw_pred = extract_pixel_point(text)

    # --- coordinate-scale correction ---
    if raw_pred is not None:
        px, py, was_normalised = maybe_denormalize(raw_pred, W, H)
        pred = (px, py)
    else:
        pred = None
        was_normalised = False

    x1, y1, x2, y2 = gt_bbox

    if pred is not None:
        px, py = pred

        # hit
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True, "hit", pred, text, was_normalised

        # Patch overlap
        patch_px = min(W, H) * PATCH_FRAC
        patch = [px - patch_px, py - patch_px, px + patch_px, py + patch_px]
        if boxes_overlap(patch, gt_bbox):
            return True, "hit_overlap", pred, text, was_normalised

    failure = classify_failure(pred, gt_bbox, W, H)
    return False, failure, pred, text, was_normalised

# Weighted prefix score helpers

def max_weighted_score(num_steps):
    """Maximum possible weighted prefix score for a task with `num_steps` steps."""
    return sum(EARLY_STEP_ALPHA ** i for i in range(num_steps))


# Task evaluation

def evaluate_task(task, model, processor, dataset_root):
    steps     = task.get("steps", [])
    num_steps = len(steps)

    correct           = 0
    weighted          = 0.0
    evaluated_steps   = 0       
    logs              = []
    task_failed       = False

    for idx, step in enumerate(steps, start=1):
        img_path = resolve_image_path(dataset_root, step["image_path"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logs.append({
                "step_id":        idx,
                "hit":            0,
                "failure_type":   "corrupt_image",
                "difficulty":     "unknown",
                "gt_bbox_area":   None,
                "pred_point":     None,
                "was_normalised": None,
                "raw_output":     None,
                "image_path":     step["image_path"],
                "error":          str(e),
            })
            task_failed = True
            break

        
        gt_action = None
        for a in step.get("actions", []):
            if a.get("type") == "click" and "bbox" in a:
                gt_action = a
                break

        
        if gt_action is None:
            logs.append({
                "step_id":        idx,
                "hit":            None,
                "failure_type":   "skipped_non_click",
                "difficulty":     "unknown",
                "gt_bbox_area":   None,
                "pred_point":     None,
                "was_normalised": None,
                "raw_output":     None,
                "image_path":     step["image_path"],
            })
            continue

        gt_bbox = percent_bbox_xywh_to_pixel_xyxy(gt_action["bbox"], *image.size)
        area    = bbox_area_ratio(gt_bbox, *image.size)
        evaluated_steps += 1

        hit, reason, pred, raw, was_normalised = evaluate_step(
            model, processor, image, step["instruction"], gt_bbox
        )

        logs.append({
            "step_id":        idx,
            "hit":            int(hit),
            "failure_type":   None if hit else reason,
            "difficulty":     difficulty_tag(area),
            "gt_bbox_area":   area,
            "pred_point":     list(pred) if pred else None,
            "was_normalised": was_normalised,
            "raw_output":     raw,
            "image_path":     step["image_path"],
        })

        if not hit:
            task_failed = True
            break

        correct  += 1
        weighted += EARLY_STEP_ALPHA ** (idx - 1)

   
    max_w = max_weighted_score(num_steps) if num_steps > 0 else 1.0
    normalised_weighted = weighted / max_w

    return {
        "task_overview":             task.get("task_overview", ""),
        "completed":                 int(correct == num_steps and not task_failed),
        "correct_steps":             correct,
        "num_steps":                 num_steps,
        "evaluated_steps":           evaluated_steps,
        "weighted_prefix_score":     weighted,
        "normalised_weighted_score": normalised_weighted,
        "steps":                     logs,
    }



# Metrics


def compute_metrics(results):
    total           = len(results)
    completed       = sum(r["completed"] for r in results)
    total_steps     = sum(r["num_steps"] for r in results)
    correct_steps   = sum(r["correct_steps"] for r in results)

  
    evaluated_steps = sum(r.get("evaluated_steps", r["num_steps"]) for r in results)

    step1_hits = sum(
        1 for r in results
        if r["steps"] and r["steps"][0].get("hit") == 1
    )

    return {
        "task_completion_accuracy":       completed / total if total else 0.0,
        "step1_accuracy":                 step1_hits / total if total else 0.0,
        "avg_correct_prefix":             correct_steps / total if total else 0.0,
        # denominator = steps that were actually evaluated (early-stop aware)
        "step_hit_rate":                  correct_steps / evaluated_steps if evaluated_steps else 0.0,
        # raw weighted (for backward compat)
        "avg_weighted_prefix_score":
            sum(r["weighted_prefix_score"] for r in results) / total if total else 0.0,
      
        "avg_normalised_weighted_score":
            sum(r["normalised_weighted_score"] for r in results) / total if total else 0.0,
        "total_tasks":                    total,
        "total_steps":                    total_steps,
        "evaluated_steps":                evaluated_steps,
    }


def failure_statistics(results):
    counter       = Counter()
    by_difficulty = defaultdict(Counter)

    for r in results:
        if not r["completed"] and r["steps"]:
            # find the first failing step
            failing = next(
                (s for s in r["steps"] if s.get("hit") == 0),
                None,
            )
            if failing:
                counter[failing["failure_type"]] += 1
                by_difficulty[failing["difficulty"]][failing["failure_type"]] += 1

    return {
        "failure_type_histogram":  dict(counter),
        "failure_by_difficulty":   {k: dict(v) for k, v in by_difficulty.items()},
    }



# Checkpointing

def load_checkpoint(ckpt_path):
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            data = json.load(f)
        completed_ids = {r["task_overview"] for r in data}
        print(f"Resuming: {len(data)} tasks already done.")
        return data, completed_ids
    return [], set()


def save_checkpoint(ckpt_path, results):
    with open(ckpt_path, "w") as f:
        json.dump(results, f, indent=2)



# Main
"""
Run:
    python evaluate_tars.py \
        --model_name_or_path ByteDance-Seed/UI-TARS-7B-SFT \
        --data_dir /path/to/Annotations/folder \
        --dataset_root /path/to/dataset_root
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--data_dir",           required=True)
    parser.add_argument("--dataset_root",       required=True)
    parser.add_argument("--resume",             action="store_true",
                        help="Resume from existing checkpoints")
    args = parser.parse_args()
    args.dataset_root = Path(args.dataset_root)

    REPO_ROOT    = Path(__file__).resolve().parents[2]
    RESULTS_ROOT = REPO_ROOT / "Results" / "UI-TARS"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.model_name_or_path)

    all_results = []

    for json_file in sorted(Path(args.data_dir).glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

       
        sanity_check_images(data["tasks"], args.dataset_root)

        run_dir = RESULTS_ROOT / json_file.stem
        run_dir.mkdir(parents=True, exist_ok=True)

  
        ckpt_path = run_dir / "checkpoint.json"
        if args.resume:
            results, done_ids = load_checkpoint(ckpt_path)
            tasks_to_run = [
                t for t in data["tasks"]
                if t.get("task_overview", "") not in done_ids
            ]
        else:
            results   = []
            tasks_to_run = data["tasks"]

        failures = [r for r in results if not r["completed"]]

        for task in tqdm(tasks_to_run, desc=json_file.stem):
            r = evaluate_task(task, model, processor, args.dataset_root)
            results.append(r)
            if not r["completed"]:
                failures.append(r)

            
            save_checkpoint(ckpt_path, results)


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

        print(f"\nFinished {json_file.name}")
        print(json.dumps(metrics, indent=2))

    overall_metrics = compute_metrics(all_results)
    with open(RESULTS_ROOT / "overall_dataset_metrics.json", "w") as f:
        json.dump(overall_metrics, f, indent=2)

    print("\nOVERALL DATASET METRICS")
    print(json.dumps(overall_metrics, indent=2))