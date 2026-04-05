"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Qwen2.5-VL AMR — PART 4: Per-Dataset Plate Image Viewer                ║
║  Dataset: https://datadryad.org/dataset/doi:10.5061/dryad.5dv41nsfj     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Run AFTER Part 1 (preprocessing) has generated image_index.csv.        ║
║  Can also run standalone — uses simulated predictions when no CSV exists.║
╠══════════════════════════════════════════════════════════════════════════╣
║  Generates per-dataset figures showing how Qwen detects each class       ║
║  across Single / Dual / Triple task variants:                            ║
║                                                                          ║
║  For EACH of the 4 datasets (ESBL, AmpC, Combination, Carbapenemase):   ║
║    Figure A — 3-row plate grid                                           ║
║               Rows = Single | Dual | Triple                              ║
║               Cols = up to 8 sample plates from that class               ║
║               Green border = correct  |  Red border = wrong             ║
║    Figure B — Horizontal bar chart: Recall / Precision / F1 per model   ║
║                                                                          ║
║  Plus one global figure:                                                 ║
║    Figure C — Heatmap: all 4 datasets × all 3 models × Recall + F1      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Outputs → data/qwen_per_dataset_results/                                ║
║    plates_esbl_all_models.png                                            ║
║    plates_ampc_all_models.png                                            ║
║    plates_combination_all_models.png                                     ║
║    plates_carbapenemase_all_models.png                                   ║
║    metrics_esbl.png  ...  metrics_carbapenemase.png                      ║
║    summary_heatmap_all_datasets.png                                      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
IMAGE_DIR       = Path("/kaggle/input/datasets/abedhossain/dryad-image-dataset-of-amr/images_original")
IMAGE_INDEX     = Path("data/processed/image_index.csv")
OUT_DIR         = Path("data/qwen_per_dataset_results")

# Set to your saved predictions CSV if available (from Part 2).
# If None, predictions are simulated from recall numbers in the training log.
PREDICTIONS_CSV = None   # e.g. Path("data/qwen_triple_results/all_predictions.csv")

MECH_CLASSES = ["ESBL", "AmpC", "Combination", "Carbapenemase"]
MODELS       = ["single", "dual", "triple"]

MODEL_LABELS = {
    "single": "Single-Task\n(mechanism only)",
    "dual":   "Dual-Task\n(+ species)",
    "triple": "Triple-Task\n(+ species + zone)",
}
MODEL_COLORS = {"single": "#378ADD", "dual": "#1D9E75", "triple": "#7F77DD"}

CLASS_COLORS = {
    "ESBL":          "#1D9E75",
    "AmpC":          "#378ADD",
    "Carbapenemase": "#E24B4A",
    "Combination":   "#7F77DD",
}

CORRECT_BORDER = "#1D9E75"
WRONG_BORDER   = "#E24B4A"
BG             = "#f5f5f2"

# Recall numbers from training log — update if your run differs
RECALL = {
    "single":  {"AmpC": .50, "Carbapenemase": .30, "Combination": .50, "ESBL": .91},
    "dual":    {"AmpC": .92, "Carbapenemase": .70, "Combination": .75, "ESBL": .90},
    "triple":  {"AmpC": 1.0, "Carbapenemase": .70, "Combination": .81, "ESBL": .98},
}
# Precision numbers from training log
PRECISION = {
    "single":  {"AmpC": .67, "Carbapenemase": .50, "Combination": .57, "ESBL": .88},
    "dual":    {"AmpC": .88, "Carbapenemase": .64, "Combination": .71, "ESBL": .93},
    "triple":  {"AmpC": .96, "Carbapenemase": .78, "Combination": .87, "ESBL": .97},
}

def f1_score(r, p):
    return round(2 * r * p / (r + p + 1e-9), 2)


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════
def load_image_index() -> pd.DataFrame:
    if IMAGE_INDEX.exists():
        df = pd.read_csv(IMAGE_INDEX)
        df = df[df["original_path"].notna()].copy().reset_index(drop=True)
        print(f"  Loaded image index: {len(df)} rows")
        return df

    print(f"  image_index.csv not found — scanning {IMAGE_DIR}")
    records = []
    if IMAGE_DIR.exists():
        for f in sorted(IMAGE_DIR.glob("*.jpg")):
            sid = f.stem.replace(" original", "").strip().rstrip(".")
            records.append({"SampleID": sid, "original_path": str(f)})
    return pd.DataFrame(records)


def build_predictions(img_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load saved predictions or simulate from training log recall numbers.
    Simulated data reproduces:
      Single-task: acc=75.6%  AmpC=50%  Carb=30%  Combo=50%  ESBL=91%
      Dual-task:   acc=87.1%  AmpC=92%  Carb=70%  Combo=75%  ESBL=90%
      Triple-task: acc=94.6%  AmpC=100% Carb=70%  Combo=81%  ESBL=98%
    """
    if PREDICTIONS_CSV and Path(PREDICTIONS_CSV).exists():
        print(f"  Loading predictions from {PREDICTIONS_CSV}")
        return pd.read_csv(PREDICTIONS_CSV)

    print("  Simulating predictions from training log recall numbers...")
    np.random.seed(42)
    CONF_CORRECT = {"single": (.60,.88), "dual": (.70,.95), "triple": (.80,.99)}
    CONF_WRONG   = {"single": (.40,.75), "dual": (.45,.80), "triple": (.50,.85)}

    if "ResistanceMechanism" in img_df.columns:
        true_mechs = img_df["ResistanceMechanism"].tolist()
        sample_ids = img_df["SampleID"].tolist()
        orig_paths = img_df["original_path"].tolist()
    else:
        # Fallback: use known dataset distribution
        dist = (["ESBL"]*97 + ["AmpC"]*24 + ["Combination"]*16 +
                ["Carbapenemase"]*10)
        np.random.shuffle(dist)
        dist       = dist[:len(img_df)]
        true_mechs = dist
        sample_ids = img_df["SampleID"].tolist()
        orig_paths = img_df["original_path"].tolist()

    records = []
    for sid, true, path in zip(sample_ids, true_mechs, orig_paths):
        if true not in MECH_CLASSES:
            continue
        row    = {"SampleID": sid, "true_mech": true, "original_path": path}
        others = [c for c in MECH_CLASSES if c != true]
        for model in MODELS:
            correct = np.random.random() < RECALL[model][true]
            pred    = true if correct else np.random.choice(others)
            lo, hi  = CONF_CORRECT[model] if correct else CONF_WRONG[model]
            conf    = round(np.random.uniform(lo, hi) * 100, 1)
            row[f"{model}_pred"]    = pred
            row[f"{model}_conf"]    = conf
            row[f"{model}_correct"] = correct
        records.append(row)

    df = pd.DataFrame(records)
    for m in MODELS:
        acc = df[f"{m}_correct"].mean()
        print(f"    {m:8s}: {acc*100:.1f}% simulated accuracy")
    return df


# ══════════════════════════════════════════════════════════════════════════
# IMAGE HELPERS
# ══════════════════════════════════════════════════════════════════════════
def load_plate(path) -> Image.Image:
    try:
        p = Path(str(path))
        if p.exists():
            return Image.open(p).convert("RGB")
    except Exception:
        pass
    return None


def make_placeholder(cls: str, size=(200, 200)) -> Image.Image:
    """Synthetic plate-like placeholder when image not found on disk."""
    arr  = np.ones((size[1], size[0], 3), dtype=np.uint8) * 20
    cx, cy = size[0]//2, size[1]//2
    yi, xi = np.ogrid[:size[1], :size[0]]
    for r in [25, 50, 75, 90]:
        m = np.abs(np.sqrt((xi-cx)**2+(yi-cy)**2) - r) < 2.5
        arr[m] = [50, 60, 50]
    return Image.fromarray(arr)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE A — per-dataset plate grid (rows = models, cols = samples)
# ══════════════════════════════════════════════════════════════════════════
def fig_per_dataset_plates(pred_df: pd.DataFrame, cls: str,
                            max_samples: int = 8,
                            save_path=None):
    """
    Rows = Single / Dual / Triple
    Columns = up to max_samples plates whose true class == cls
    Green border = correct  |  Red border = wrong
    """
    cls_df  = pred_df[pred_df["true_mech"] == cls].head(max_samples)
    samples = cls_df.to_dict("records")
    n       = len(samples)
    if n == 0:
        print(f"  No samples found for {cls}")
        return

    n_total = len(pred_df[pred_df["true_mech"] == cls])
    fig, axes = plt.subplots(
        3, n, figsize=(n * 2.5, 3 * 3.0 + 1.2), facecolor=BG)

    # Ensure axes is always 2D
    if n == 1:
        axes = [[axes[ri]] for ri in range(3)]

    fig.suptitle(
        f"Dataset: {cls}  (n={n_total} total plates)\n"
        f"Rows = task model  |  Green border = correct  |  Red border = wrong",
        fontsize=11, fontweight="bold", color="#1a1d1a", y=1.02)

    for ri, model in enumerate(MODELS):
        axes[ri][0].set_ylabel(
            MODEL_LABELS[model], fontsize=9, rotation=90, labelpad=4,
            color=MODEL_COLORS[model], fontweight="bold")

    for ci, sample in enumerate(samples):
        img = load_plate(sample["original_path"]) or make_placeholder(cls)

        for ri, model in enumerate(MODELS):
            ax      = axes[ri][ci]
            pred    = sample[f"{model}_pred"]
            conf    = sample[f"{model}_conf"]
            correct = sample[f"{model}_correct"]
            sid     = sample["SampleID"]

            ax.imshow(img, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])

            border = CORRECT_BORDER if correct else WRONG_BORDER
            symbol = "✓" if correct else "✗"
            for spine in ax.spines.values():
                spine.set_edgecolor(border); spine.set_linewidth(4)

            if ri == 0:
                ax.set_title(f"ID: {sid}", fontsize=7, color="#666", pad=3)

            y = 1.01
            for txt, col, sz in [
                (f"{symbol} {cls}",  CORRECT_BORDER if correct else WRONG_BORDER, 8),
                (f"→ {pred}",        CLASS_COLORS.get(pred, "#888"),              8),
                (f"{conf}%",         "#555",                                       7),
            ]:
                ax.text(0.5, y, txt, transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=sz,
                        color=col, fontfamily="monospace")
                y += 0.07

            # Confidence dot
            ax.add_patch(patches.Circle(
                (0.90, 0.06), 0.05, transform=ax.transAxes,
                color=CLASS_COLORS.get(pred, "#888"), zorder=5))

    legend_handles = (
        [patches.Patch(color=CORRECT_BORDER, label="Correct"),
         patches.Patch(color=WRONG_BORDER,   label="Wrong")]
        + [patches.Patch(color=v, label=k) for k, v in CLASS_COLORS.items()]
    )
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               fontsize=8, bbox_to_anchor=(0.5, -0.04),
               framealpha=0.9, edgecolor="#ccc")

    plt.subplots_adjust(wspace=0.05, hspace=0.55, top=0.92)
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight", facecolor=BG)
        print(f"  Saved → {save_path}")
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE B — per-dataset metric bars (Recall / Precision / F1)
# ══════════════════════════════════════════════════════════════════════════
def fig_per_dataset_metrics(pred_df: pd.DataFrame, cls: str, save_path=None):
    """
    Horizontal bar chart for Recall / Precision / F1
    comparing Single, Dual, Triple models on one dataset.
    """
    metrics = {
        "Recall":    {m: RECALL[m][cls]    for m in MODELS},
        "Precision": {m: PRECISION[m][cls] for m in MODELS},
        "F1":        {m: f1_score(RECALL[m][cls], PRECISION[m][cls]) for m in MODELS},
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), facecolor=BG)
    fig.suptitle(f"Detection Metrics — {cls}",
                 fontsize=12, fontweight="bold", color="#1a1d1a")

    for ax, (metric_name, vals) in zip(axes, metrics.items()):
        model_list = list(vals.keys())
        bar_vals   = [vals[m] * 100 for m in model_list]
        colors     = [MODEL_COLORS[m] for m in model_list]

        bars = ax.barh(model_list, bar_vals, color=colors,
                       height=0.5, edgecolor="none")
        for bar, val in zip(bars, bar_vals):
            ax.text(bar.get_width() + 1,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.0f}%", va="center", fontsize=9, color="#333")

        ax.set_xlim(0, 112)
        ax.set_xlabel(metric_name + " (%)", fontsize=9)
        ax.set_facecolor(BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.axvline(50, color="#ccc", lw=0.8, ls="--")
        ax.axvline(80, color="#bbb", lw=0.8, ls="--")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight", facecolor=BG)
        print(f"  Saved → {save_path}")
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE C — global summary heatmap
# ══════════════════════════════════════════════════════════════════════════
def fig_summary_heatmap(save_path=None):
    """
    Heatmap: rows = 4 datasets, cols = 3 models × (Recall + F1)
    Red → yellow → green shows where each model performs well or struggles.
    """
    col_labels  = ([f"{m.capitalize()}\nRecall" for m in MODELS] +
                   [f"{m.capitalize()}\nF1"     for m in MODELS])
    model_keys  = MODELS + MODELS
    metric_keys = ["recall"] * 3 + ["f1"] * 3

    lookup = {
        "recall": RECALL,
        "f1": {
            m: {c: f1_score(RECALL[m][c], PRECISION[m][c]) for c in MECH_CLASSES}
            for m in MODELS
        },
    }

    data = np.array([
        [lookup[mk][m][cls] * 100 for m, mk in zip(model_keys, metric_keys)]
        for cls in MECH_CLASSES
    ])

    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor=BG)
    im = ax.imshow(data, cmap="RdYlGn", vmin=30, vmax=100, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(MECH_CLASSES)))
    ax.set_yticklabels(MECH_CLASSES, fontsize=10)
    ax.set_facecolor(BG)

    # Separator between Recall and F1 blocks
    ax.axvline(2.5, color="white", lw=2.5)

    for r in range(len(MECH_CLASSES)):
        for c in range(len(col_labels)):
            v          = data[r, c]
            text_color = "white" if v < 55 else "#1a1a1a"
            ax.text(c, r, f"{v:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Score (%)", fontsize=8)

    ax.set_title(
        "Summary: All Datasets × All Models  (Recall + F1)\n"
        "Left 3 cols = Recall  |  Right 3 cols = F1",
        fontsize=11, fontweight="bold", pad=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight", facecolor=BG)
        print(f"  Saved → {save_path}")
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE D — side-by-side comparison for hard samples
# ══════════════════════════════════════════════════════════════════════════
def fig_hard_sample_comparison(pred_df: pd.DataFrame, save_path=None):
    """
    Rows  = same 6 hard plates (wrong in single, may be correct in triple)
    Cols  = Single | Dual | Triple
    Confidence bar shown below each image.
    """
    # Pick samples wrong in single, prefer those corrected by triple
    improved   = pred_df[
        (~pred_df["single_correct"]) & (pred_df["triple_correct"])
    ]["SampleID"].tolist()
    still_wrong = pred_df[
        (~pred_df["single_correct"]) & (~pred_df["triple_correct"])
    ]["SampleID"].tolist()
    hard_ids   = (improved[:4] + still_wrong[:2])[:6]
    if len(hard_ids) < 6:
        hard_ids += pred_df[~pred_df["single_correct"]]["SampleID"].tolist()
    hard_ids   = hard_ids[:6]

    rows_data  = (pred_df[pred_df["SampleID"].isin(hard_ids)]
                  .to_dict("records"))
    id_order   = {sid: i for i, sid in enumerate(hard_ids)}
    rows_data  = sorted(rows_data,
                        key=lambda r: id_order.get(r["SampleID"], 999))

    n_plates   = len(rows_data)
    if n_plates == 0:
        print("  No hard samples found.")
        return

    fig, axes = plt.subplots(
        n_plates, 3,
        figsize=(14, n_plates * 3.8 + 1.2),
        facecolor=BG)
    fig.suptitle(
        "Hard Samples — Same Plates Compared Across All Three Models\n"
        "Green border = correct  |  Red border = wrong  "
        "|  Bar = confidence",
        fontsize=12, fontweight="bold", color="#1a1d1a", y=1.01)

    model_titles = {
        "single": "Single-Task\n(mechanism only)",
        "dual":   "Dual-Task\n(+ species)",
        "triple": "Triple-Task\n(+ species + zone)",
    }

    for col, model in enumerate(MODELS):
        axes[0][col].set_title(
            model_titles[model], fontsize=10, fontweight="bold",
            color=MODEL_COLORS[model], pad=8)

    for row_i, row in enumerate(rows_data):
        img = load_plate(row["original_path"]) or make_placeholder(
            row["true_mech"])

        for col, model in enumerate(MODELS):
            ax      = axes[row_i][col]
            pred    = row[f"{model}_pred"]
            conf    = row[f"{model}_conf"]
            correct = row[f"{model}_correct"]

            ax.imshow(img, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])

            border = CORRECT_BORDER if correct else WRONG_BORDER
            symbol = "✓" if correct else "✗"
            for spine in ax.spines.values():
                spine.set_edgecolor(border); spine.set_linewidth(5)

            label_color = "#1a5c3a" if correct else WRONG_BORDER
            lines = [
                (f"{symbol} True: {row['true_mech']}", border,       9, "bold"),
                (f"Pred: {pred}",                      label_color,  8, "normal"),
                (f"Conf: {conf}%",                     label_color,  8, "normal"),
            ]
            if col == 0:
                lines.append((f"ID: {row['SampleID']}", "#777", 7, "normal"))

            y_pos = 1.00
            for text, color, size, weight in lines:
                ax.text(0.5, y_pos, text, transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=size,
                        fontweight=weight, color=color,
                        fontfamily="monospace")
                y_pos += 0.08

            # Confidence progress bar below the image
            bar_color = CORRECT_BORDER if correct else WRONG_BORDER
            ax.add_patch(patches.FancyBboxPatch(
                (0, -0.06), conf/100, 0.04, transform=ax.transAxes,
                boxstyle="square,pad=0", color=bar_color, alpha=0.75,
                zorder=5, clip_on=False))
            ax.add_patch(patches.FancyBboxPatch(
                (0, -0.06), 1.0, 0.04, transform=ax.transAxes,
                boxstyle="square,pad=0", color="#ddd", alpha=0.3,
                zorder=4, clip_on=False))

    legend_handles = (
        [patches.Patch(color=CORRECT_BORDER, label="Correct"),
         patches.Patch(color=WRONG_BORDER,   label="Wrong")]
        + [patches.Patch(color=v, label=k) for k, v in CLASS_COLORS.items()]
    )
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               fontsize=9, bbox_to_anchor=(0.5, -0.02),
               framealpha=0.9, edgecolor="#ccc")

    plt.subplots_adjust(wspace=0.04, hspace=0.60, top=0.94)
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight", facecolor=BG)
        print(f"  Saved → {save_path}")
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def run():
    print("=" * 62)
    print("  Qwen AMR — Per-Dataset Detection Viewer")
    print("=" * 62)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_df  = load_image_index()
    pred_df = build_predictions(img_df)

    # ── Per-dataset figures ────────────────────────────────────────────────
    for cls in MECH_CLASSES:
        print(f"\n{'─'*52}")
        print(f"  Dataset: {cls}")
        print(f"{'─'*52}")

        # Figure A — plate grid
        fig_per_dataset_plates(
            pred_df, cls,
            max_samples=8,
            save_path=OUT_DIR / f"plates_{cls.lower()}_all_models.png",
        )

        # Figure B — metric bars
        fig_per_dataset_metrics(
            pred_df, cls,
            save_path=OUT_DIR / f"metrics_{cls.lower()}.png",
        )

    # ── Global summary heatmap ─────────────────────────────────────────────
    print(f"\n{'─'*52}")
    print("  Summary heatmap across all datasets")
    print(f"{'─'*52}")
    fig_summary_heatmap(
        save_path=OUT_DIR / "summary_heatmap_all_datasets.png")

    # ── Hard-sample comparison ─────────────────────────────────────────────
    print(f"\n{'─'*52}")
    print("  Hard-sample comparison (same plates, all 3 models)")
    print(f"{'─'*52}")
    fig_hard_sample_comparison(
        pred_df,
        save_path=OUT_DIR / "hard_sample_comparison.png")

    print("\n" + "=" * 62)
    print("  All figures saved to:", OUT_DIR)
    print("  Per-dataset plate grids  : plates_<class>_all_models.png")
    print("  Per-dataset metric bars  : metrics_<class>.png")
    print("  Global heatmap           : summary_heatmap_all_datasets.png")
    print("  Hard-sample comparison   : hard_sample_comparison.png")
    print("=" * 62)
    return pred_df


if __name__ == "__main__":
    pred_df = run()
