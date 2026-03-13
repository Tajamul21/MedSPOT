# MedSPOT-Bench

**MedSPOT-Bench** is a benchmark for evaluating Multimodal Large Language Models (MLLMs) on GUI grounding tasks in medical imaging software.

## Overview

MedSPOT-Bench evaluates models on their ability to localize and interact with UI elements across 10 medical imaging applications including 3DSlicer, DICOMscope, Weasis, MITK, and others.

### Metrics
- **TCA (Task Completion Accuracy)**: Fraction of tasks where ALL steps are completed correctly in sequence
- **SHR (Step Hit Rate)**: Per-step accuracy across all evaluated steps  
- **S1A (Step 1 Accuracy)**: Accuracy on the first step of each task

### Evaluation Protocol
Tasks are evaluated **sequentially** — if a model fails a step, the task is terminated early. This reflects real-world GUI interaction where errors compound.

---

## Evaluation Scripts

Each script evaluates one model on the full benchmark. All scripts share the same interface:

| Script | Model |
|--------|-------|
| `evaluate_gui_actor.py` | GUI-Actor |
| `evaluate_gpt5.py` | GPT-5 |
| `evaluate_gpt4omini.py` | GPT-4o-mini |
| `evaluate_tars.py` | UI-TARS |
| `evaluate_cogagent.py` | CogAgent-9B |
| `evaluate_qwen2vl.py` | Qwen2-VL |
| `evaluate_qwen2_5vl.py` | Qwen2.5-VL |
| `evaluate_qwen3vl0.py` | Qwen3-VL |
| `evaluate_gemma3_27B.py` | Gemma3-27B |
| `evaluate_llama.py` | Llama-3.2-11B |
| `evaluate_osatlas.py` | OS-Atlas |
| `evaluate_seeclick.py` | SeeClick |
| `evaluate_uground.py` | UGround |
| `evaluate_Aguvis.py`  | Aguvis-7B |

---

## Usage

### HuggingFace Models
```bash
python evaluate_gui_actor.py \
    --model_name_or_path <hf_model_path> \
    --data_dir           /path/to/Annotations \
    --dataset_root       /path/to/MedSPOT-Bench \
    --results_dir        /path/to/save/results
```

### OpenAI Models (GPT-5, GPT-4o-mini)
```bash
export OPENAI_API_KEY=your_api_key

python evaluate_gpt5.py \
    --data_dir     /path/to/Annotations \
    --dataset_root /path/to/MedSPOT-Bench \
    --results_dir  /path/to/save/results
```

---
## Dataset
The data will be available on HuggingFace at https://huggingface.co/datasets/anonymous009009/MedSPOT_Bench

## Dataset Structure

```
MedSPOT-Bench/
  Annotations/
    3DSlicer_Annotation.json
    DICOMscope_Annotation.json
    Weasis_Annotation.json
    ...
  Images/
    3DSlicer/
    DICOMscope/
    Weasis/
    ...
```

Each annotation JSON follows this format:
```json
{
  "tasks": [
    {
      "task_overview": "Delete the selected study from the browser menu",
      "steps": [
        {
          "step_id": 1,
          "image_path": "Images/DICOMscope/Screenshot_250.png",
          "instruction": "Select a study from the Browser menu",
          "actions": [{"type": "click", "bbox": [x%, y%, w%, h%]}]
        }
      ]
    }
  ]
}
```

---

## Requirements

Each model should be evaluated in its own recommended environment.  
Please follow the official setup instructions for each model:

Common dependencies across all environments:
- `torch >= 2.0`
- `transformers >= 4.40`
- `pillow`, `tqdm`

## Results

Results are saved in the following structure:
```
results/
  ModelName/
    SoftwareName/
      task_results.json
      task_metrics.json
      failure_statistics.json
    overall_dataset_metrics.json
```

---

## Citation

```bibtex
@article{medspot2026,
  title     = {MedSPOT: A Workflow-Aware Sequential Grounding Benchmark for Clinical GUI},
  year      = {2026}
  note      = {Under review}
}
```
