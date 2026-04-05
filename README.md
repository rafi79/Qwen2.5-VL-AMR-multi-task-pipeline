# Qwen2.5-VL-3B AMR Multi-Task Learning

> **Automated Antimicrobial Resistance Detection from Disk Diffusion Plate Images**
> Using a fine-tuned vision-language model with three simultaneous tasks.
<img width="2244" height="1519" alt="plates_esbl_all_models" src="https://github.com/user-attachments/assets/12cd4d64-34cd-41cb-ad98-f7ee9e4da4d5" />

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Dryad-orange.svg)](https://datadryad.org/dataset/doi:10.5061/dryad.5dv41nsfj)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20T4×2-yellow.svg)](https://www.kaggle.com/)

---

## Overview

This repository fine-tunes **Qwen2.5-VL-3B-Instruct** on disk diffusion plate photographs
to classify bacterial resistance mechanisms (ESBL, AmpC, Combination, Carbapenemase).
Three progressively richer task variants are compared:

| Model | Tasks | Val Accuracy | Weighted F1 | Cohen κ |
|-------|-------|:---:|:---:|:---:|
| Single-task | Mechanism only | 75.6% ± 9.0% | 0.721 ± 0.130 | 0.440 |
| Dual-task | + Species auxiliary | 87.1% ± 3.6% | 0.874 ± 0.034 | 0.761 |
| **Triple-task** | **+ Zone regression** | **94.6% ± 1.8%** | **0.944 ± 0.018** | **0.897** |
| CLIP (baseline) | Mechanism only | 81.0% | 0.789 | 0.628 |

---

## Dataset

**Source:** [Dryad — Image Dataset of AMR](https://datadryad.org/dataset/doi:10.5061/dryad.5dv41nsfj)

```
/kaggle/input/datasets/abedhossain/dryad-image-dataset-of-amr/
  ├── images_original/   ← 147 disk diffusion plate photos (.jpg)
  ├── images_measured/   ← same plates with zone annotations
  └── Tables/
      ├── Overview GPT JCM.xlsx   ← resistance mechanism labels
      └── Table X.X.X..docx       ← per-antibiotic zone + S/I/R
```

| Statistic | Value |
|-----------|-------|
| Total images | 147 |
| Resistance classes | 4 (ESBL 97, AmpC 24, Combination 16, Carbapenemase 10) |
| Species | 10 |
| Antibiotics tracked | 23 |
| Zone matrix coverage | 67.8% |

---

## Repository Structure

```
qwen_amr_project/
  01_preprocessing.py         ← Parse xlsx + docx, build image_index.csv
  02_train_multitask.py       ← Qwen2.5-VL LoRA training (OOM-fixed for T4×2)
  03_visualize_results.py     ← 8 publication-quality result plots
  04_per_dataset_viewer.py    ← Per-class plate grids + heatmap viewer
  README.md                   ← This file
```

---

## Pipeline — Run in Order

### 0. Install dependencies

```bash
pip install einops timm -q
pip install python-docx openpyxl -q
pip install bitsandbytes accelerate -q
pip install peft einops timm -q
pip install peft --quiet
```

---

### 1. Preprocessing — `01_preprocessing.py`

Parses the Dryad Excel + Word files and builds clean CSVs:

```python
%run 01_preprocessing.py
```

**Outputs:**
```
data/processed/
  master_labels.csv   (one row per antibiotic record)
  image_index.csv     (one row per plate image)
  class_distribution.png
```

**Cleaning steps applied:**
- Drop rows with missing plate images
- Drop rows with missing S/I/R interpretation
- Drop zone diameters outside [6, 50] mm
- Standardise species names (18 variant → 10 canonical)
- Derive 4-class `ResistanceMechanism` from ESBL / AmpC / Carbapenemase boolean flags

---

### 2. Multi-Task Training — `02_train_multitask.py`

Runs 5-fold stratified CV for Single-task, Dual-task, and Triple-task variants:

```python
# Set your HuggingFace token as a Kaggle Secret or environment variable:
import os
os.environ["HF_TOKEN"] = "hf_..."

%run 02_train_multitask.py
```

**OOM fixes for T4×2 (15.6 GB each):**

| Fix | Description | VRAM saved |
|-----|-------------|------------|
| 1 | Visual encoder frozen | ~3 GB |
| 2 | MAX_PIXELS halved (256→128 × 28²) | ~1 GB |
| 3 | max_length 512→256 | ~1 GB |
| 4 | LoRA rank 16→8 | ~0.5 GB |
| 5 | `use_cache=False` in training forward pass | ~0.5 GB |
| 6 | `PYTORCH_ALLOC_CONF=expandable_segments:True` | fragmentation |
| 7 | batch=2 + grad_accum=8 (effective=16) | peak VRAM halved |
| 8 | MAX_MEMORY=12 GiB per GPU | headroom |
| 9 | Gradient checkpointing on LM decoder | ~1 GB |

**Architecture:**
```
Qwen2.5-VL-3B backbone (LoRA r=8, only LM decoder)
  └── Last-token pooled hidden state [batch, 2048]
        ├── MechanismHead    → 4-class logits   (CrossEntropy, weighted)
        ├── SpeciesHead      → N-class logits   (CrossEntropy, λ=0.4)
        └── ZoneHead         → 23-dim regression (SmoothL1, λ=0.3)

Loss = mech_loss + 0.4 × species_loss + 0.3 × zone_loss
```

**Outputs:**
```
data/qwen_triple_results/
  cv_single.csv  |  cv_dual.csv  |  cv_triple.csv
  all_histories.csv
  1_three_way_comparison.png
  2_confusion_trio.png
  3_per_class_f1.png
  4_training_curves.png
  5_qwen_vs_clip.png
```

---

### 3. Result Visualizations — `03_visualize_results.py`

Generates 8 publication-quality figures from hardcoded training log numbers.
**No re-training needed** — runs instantly:

```python
%run 03_visualize_results.py
```

| Figure | Description |
|--------|-------------|
| `viz_01_dataset_overview.png` | Mechanism/species distribution + key stats |
| `viz_02_per_sample_grid.png` | All 147 plates across all 3 models |
| `viz_03_fold_training_journey.png` | Per-fold val accuracy + F1 at each checkpoint |
| `viz_04_confusion_matrices.png` | 5-fold pooled confusion matrices |
| `viz_05_per_class_deep_dive.png` | Per-class Precision / Recall / F1 |
| `viz_06_fold_variance.png` | Stability across folds + CLIP comparison line |
| `viz_07_zone_regression.png` | Zone MAE per fold + antibiotic coverage heatmap |
| `viz_08_final_summary.png` | All models vs CLIP baseline |

---

### 4. Per-Dataset Viewer — `04_per_dataset_viewer.py`

Detailed per-class analysis showing how Qwen detects each dataset:

```python
%run 04_per_dataset_viewer.py
```

For each of the 4 resistance classes:

- **Figure A** — Plate grid: rows = Single/Dual/Triple, columns = individual samples.
  Green border = correct, red = wrong. Confidence score and prediction label overlaid.
- **Figure B** — Horizontal bars: Recall / Precision / F1 for all three models.

Plus:
- **Figure C** — Summary heatmap: 4 datasets × 3 models × (Recall + F1) in a single view.
- **Figure D** — Hard-sample comparison: same 6 plates that were wrong in single-task,
  shown across all three models to visualize the multi-task gain.

**Outputs:**
```
data/qwen_per_dataset_results/
  plates_esbl_all_models.png
  plates_ampc_all_models.png
  plates_combination_all_models.png
  plates_carbapenemase_all_models.png
  metrics_esbl.png
  metrics_ampc.png
  metrics_combination.png
  metrics_carbapenemase.png
  summary_heatmap_all_datasets.png
  hard_sample_comparison.png
```

---

## Key Results

### Per-class F1 (5-fold pooled)

| Class | Single-task | Dual-task | Triple-task |
|-------|:-----------:|:---------:|:-----------:|
| ESBL (n=97) | 0.83 | 0.92 | **0.98** |
| AmpC (n=24) | 0.62 | 0.94 | **0.96** |
| Combination (n=16) | 0.53 | 0.69 | **0.79** |
| Carbapenemase (n=10) | 0.43 | 0.64 | **0.78** |

### Zone regression (Triple-task only)

| Metric | Value |
|--------|-------|
| Mean Absolute Error | 4.5 ± 0.3 mm |
| Clinical threshold | < 6 mm |
| All 5 folds below threshold | ✓ |

### Species accuracy (Triple-task)

60.6% on 10-class species identification as an auxiliary task.

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-VL-3B-Instruct |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| LoRA target modules | q/k/v/o_proj, gate/up/down_proj |
| Trainable params | 0.397% of total |
| Learning rate | 2e-4 (AdamW) |
| Weight decay | 1e-4 |
| Batch size | 2 (grad accum 8 → effective 16) |
| Epochs | 15 |
| LR schedule | CosineAnnealingLR |
| CV | 5-fold stratified |
| Image resolution | MAX_PIXELS = 128×28×28 |
| Max sequence length | 256 tokens |

## License

MIT License — see [LICENSE](LICENSE)
