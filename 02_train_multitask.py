"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Qwen2.5-VL AMR — PART 2: Triple Multi-Task Training (OOM-Fixed T4×2)   ║
║  Dataset: https://datadryad.org/dataset/doi:10.5061/dryad.5dv41nsfj     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Run AFTER Part 1 (01_preprocessing.py) has generated:                  ║
║    data/processed/master_labels.csv                                      ║
║    data/processed/image_index.csv                                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  OOM Fixes applied for Kaggle T4×2 (15.6 GB each):                     ║
║    Fix 1: Visual encoder fully frozen  → saves ~3 GB VRAM               ║
║    Fix 2: MAX_PIXELS halved (256→128 × 28×28)                           ║
║    Fix 3: max_length 512 → 256  → halves KV-cache                       ║
║    Fix 4: LoRA rank 16 → 8  → halves LoRA activation VRAM               ║
║    Fix 5: use_cache=False during training forward pass                   ║
║    Fix 6: PYTORCH_ALLOC_CONF=expandable_segments:True                   ║
║    Fix 7: batch_size=2 + grad_accum=8 (effective batch=16)              ║
║    Fix 8: MAX_MEMORY tightened to 12 GiB per GPU                        ║
║    Fix 9: gradient checkpointing on LM decoder layers                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Three tasks sharing one Qwen2.5-VL-3B backbone:                        ║
║    Task 1 (PRIMARY)   : Resistance mechanism (4 classes) → CrossEntropy ║
║    Task 2 (AUXILIARY) : Species prediction  (N classes)  → CrossEntropy ║
║    Task 3 (AUXILIARY) : Zone diameter regression (mm)    → SmoothL1    ║
║  Loss: total = mech_loss + 0.4×species_loss + 0.3×zone_loss             ║
║  CV  : 5-fold stratified (same structure as CLIP pipeline)              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Outputs → data/qwen_triple_results/                                     ║
║    cv_single.csv  |  cv_dual.csv  |  cv_triple.csv                      ║
║    all_histories.csv                                                     ║
║    1_three_way_comparison.png  through  5_qwen_vs_clip.png               ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import gc
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from collections import Counter

# ── OOM Fix 6: memory allocator must be set before importing torch ────────
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ── HuggingFace token ─────────────────────────────────────────────────────
# Replace with your own token from https://huggingface.co/settings/tokens
HF_TOKEN = os.environ.get("HF_TOKEN", "")   # set as Kaggle Secret
os.environ["HF_TOKEN"] = HF_TOKEN

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, cohen_kappa_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

# ── Paths ─────────────────────────────────────────────────────────────────
IMAGE_INDEX = Path("data/processed/image_index.csv")
MASTER_CSV  = Path("data/processed/master_labels.csv")
OUT_DIR     = Path("data/qwen_triple_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hardware ──────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPUS = torch.cuda.device_count()
DTYPE  = torch.float16   # T4 has no bfloat16 CUDA support

# OOM Fix 8: tighter cap — leaves 3.5 GiB headroom on each T4
MAX_MEMORY        = {i: "12GiB" for i in range(N_GPUS)}
MAX_MEMORY["cpu"] = "20GiB"

MODEL_NAME   = "Qwen/Qwen2.5-VL-3B-Instruct"
MECH_CLASSES = ["AmpC", "Carbapenemase", "Combination", "ESBL"]

LAMBDA_SPECIES = 0.4
LAMBDA_ZONE    = 0.3

# OOM Fix 2: smaller image tiles
MIN_PIXELS = 64  * 28 * 28    # 50176
MAX_PIXELS = 128 * 28 * 28    # 100352

CLASS_COLORS = {
    "ESBL":          "#1D9E75",
    "AmpC":          "#378ADD",
    "Carbapenemase": "#E24B4A",
    "Combination":   "#7F77DD",
}

print(f"Device: {DEVICE}  |  GPUs: {N_GPUS}  |  Model: {MODEL_NAME}")
for i in range(N_GPUS):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}  ({p.total_memory/1e9:.1f} GB)")


# ══════════════════════════════════════════════════════════════════════════
# SPECIES NORMALISATION MAP
# ══════════════════════════════════════════════════════════════════════════
SPECIES_MAP = {
    "E. coli": "Escherichia coli",
    "Escherichia coli": "Escherichia coli",
    "Escherichia coli ": "Escherichia coli",
    "K. pneumoniae": "Klebsiella pneumoniae",
    "Klebsiella pneumoniae": "Klebsiella pneumoniae",
    "E. cloacae": "Enterobacter cloacae",
    "Enterobacter cloacae": "Enterobacter cloacae",
    "Enterobacter cloacae-Komplex": "Enterobacter cloacae",
    "Klebsiella aerogenes": "Klebsiella aerogenes",
    "Klebsiella oxytoca": "Klebsiella oxytoca",
    "Proteus mirabilis": "Proteus mirabilis",
    "Serratia marcescens": "Serratia marcescens",
    "Morganella morganii": "Morganella morganii",
    "Citrobacter freundii": "Citrobacter freundii",
    "Citrobacter koseri": "Citrobacter koseri",
    "Acinetobacter baumannii": "Acinetobacter baumannii",
    "Pseudomonas aeruginosa": "Pseudomonas aeruginosa",
    "Proteus vulgaris": "Proteus vulgaris",
}


# ══════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════
def build_zone_matrix(master_df, image_df):
    """Build [n_images × n_antibiotics] normalised zone matrix."""
    ab_list     = sorted(master_df["Abbreviation"].dropna().unique().tolist())
    n_ab        = len(ab_list)
    ab2col      = {ab: i for i, ab in enumerate(ab_list)}
    sids        = image_df["SampleID"].tolist()
    n_images    = len(sids)
    zone_matrix = np.full((n_images, n_ab), np.nan, dtype=np.float32)
    sid2row     = {sid: i for i, sid in enumerate(sids)}

    for _, row in master_df.iterrows():
        sid = row["SampleID"]
        ab  = row["Abbreviation"]
        z   = row["ZoneDiameter_mm"]
        if sid in sid2row and ab in ab2col and not np.isnan(float(z)):
            zone_matrix[sid2row[sid], ab2col[ab]] = float(z)

    coverage = (~np.isnan(zone_matrix)).mean() * 100
    print(f"\nZone matrix: {n_images} × {n_ab}  |  coverage: {coverage:.1f}%")

    zone_mean = float(np.nanmean(zone_matrix))
    zone_std  = float(np.nanstd(zone_matrix))
    zone_norm = (zone_matrix - zone_mean) / (zone_std + 1e-6)
    zone_norm[np.isnan(zone_matrix)] = np.nan

    return zone_norm, zone_matrix, ab_list, ab2col, sid2row, zone_mean, zone_std


def load_data():
    """Load and merge image_index + master_labels, build zone matrix."""
    img_df    = pd.read_csv(IMAGE_INDEX)
    master_df = pd.read_csv(MASTER_CSV)

    img_df = img_df[
        img_df["original_path"].notna() &
        img_df["ResistanceMechanism"].isin(MECH_CLASSES)
    ].copy().reset_index(drop=True)

    master_df = master_df[
        master_df["SampleID"].isin(img_df["SampleID"]) &
        master_df["Abbreviation"].notna() &
        master_df["ZoneDiameter_mm"].notna()
    ].copy()

    # Clean species names
    img_df["SPECIES_CLEAN"] = (
        img_df["SPECIES"].map(SPECIES_MAP).fillna(img_df["SPECIES"]))
    counts = img_df["SPECIES_CLEAN"].value_counts()
    keep   = counts[counts >= 3].index
    img_df["SPECIES_CLEAN"] = img_df["SPECIES_CLEAN"].where(
        img_df["SPECIES_CLEAN"].isin(keep), "Other")

    le_species        = LabelEncoder()
    img_df["spec_id"] = le_species.fit_transform(img_df["SPECIES_CLEAN"])
    mech2id           = {m: i for i, m in enumerate(MECH_CLASSES)}
    img_df["mech_id"] = img_df["ResistanceMechanism"].map(mech2id)
    n_species         = img_df["SPECIES_CLEAN"].nunique()

    (zone_norm, zone_raw, ab_list, ab2col,
     sid2row, zone_mean, zone_std) = build_zone_matrix(master_df, img_df)

    print(f"\nDataset: {len(img_df)} images  |  "
          f"{len(MECH_CLASSES)} mechanisms  |  "
          f"{n_species} species  |  {len(ab_list)} antibiotics")
    print("\nMechanism distribution:")
    print(img_df["ResistanceMechanism"].value_counts().to_string())

    return {
        "img_df":    img_df,
        "master_df": master_df,
        "le_species": le_species,
        "mech2id":   mech2id,
        "id2mech":   {i: m for m, i in mech2id.items()},
        "n_species": n_species,
        "zone_norm": zone_norm,
        "zone_raw":  zone_raw,
        "ab_list":   ab_list,
        "ab2col":    ab2col,
        "sid2row":   sid2row,
        "zone_mean": zone_mean,
        "zone_std":  zone_std,
    }


# ══════════════════════════════════════════════════════════════════════════
# BACKBONE LOADER
# ══════════════════════════════════════════════════════════════════════════
def load_qwen_backbone():
    token = os.environ.get("HF_TOKEN")
    print(f"\nLoading {MODEL_NAME} ...")

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
        max_memory=MAX_MEMORY,
        token=token,
        ignore_mismatched_sizes=True,
        # No flash_attention_2 — T4 is Turing (pre-Ampere)
    )

    # OOM Fix 9: gradient checkpointing on LM decoder
    if hasattr(base, "model") and hasattr(
            base.model, "gradient_checkpointing_enable"):
        base.model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")

    # OOM Fix 1: freeze ENTIRE visual encoder
    for name, param in base.named_parameters():
        if "visual" in name:
            param.requires_grad = False
    frozen_v = sum(p.numel() for n, p in base.named_parameters()
                   if "visual" in n)
    print(f"Visual encoder frozen: {frozen_v/1e6:.0f}M params")

    # OOM Fix 4: LoRA rank 8 — target LM decoder only, NOT visual encoder
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=None,
        layers_to_transform=None,
    )

    model = get_peft_model(base, lora_cfg)

    # Double-check: ensure visual encoder stayed frozen after PEFT wrapping
    for name, param in model.named_parameters():
        if "visual" in name and param.requires_grad:
            param.requires_grad = False

    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        token=token,
        use_fast=False,
    )

    print("Backbone + LoRA ready.")
    return model, processor


# ══════════════════════════════════════════════════════════════════════════
# TASK HEADS
# ══════════════════════════════════════════════════════════════════════════
class MechanismHead(nn.Module):
    """Primary classification head: resistance mechanism (4 classes)."""
    def __init__(self, hidden: int, n_mech: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden), nn.Dropout(0.3),
            nn.Linear(hidden, 256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, n_mech),
        )
    def forward(self, x): return self.net(x)


class SpeciesHead(nn.Module):
    """Auxiliary head: species classification (N classes)."""
    def __init__(self, hidden: int, n_species: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden), nn.Dropout(0.3),
            nn.Linear(hidden, 256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, n_species),
        )
    def forward(self, x): return self.net(x)


class ZoneHead(nn.Module):
    """Auxiliary head: zone diameter regression (n_ab outputs in mm)."""
    def __init__(self, hidden: int, n_ab: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden), nn.Dropout(0.2),
            nn.Linear(hidden, 128), nn.GELU(),
            nn.Linear(128, n_ab),
        )
    def forward(self, x): return self.net(x)


# ══════════════════════════════════════════════════════════════════════════
# FULL MULTI-TASK MODEL
# ══════════════════════════════════════════════════════════════════════════
class QwenMultiTaskModel(nn.Module):
    """
    Qwen2.5-VL backbone (LoRA on LM decoder, visual encoder frozen)
    + three task heads attached to the last-token pooled hidden state.

    Pooling strategy:
      Run forward with output_hidden_states=True.
      Take hidden_states[-1] at the last real token position.
      Upcast to float32 for stable head computation.
    """

    def __init__(self, backbone, hidden: int,
                 n_mech: int, n_species: int, n_ab: int,
                 model_type: str = "triple"):
        super().__init__()
        self.backbone   = backbone
        self.model_type = model_type
        self.hidden     = hidden

        head_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.mech_head = MechanismHead(hidden, n_mech).to(head_device)

        self.species_head = (
            SpeciesHead(hidden, n_species).to(head_device)
            if model_type in ("dual", "triple") else None)

        self.zone_head = (
            ZoneHead(hidden, n_ab).to(head_device)
            if model_type == "triple" else None)

    def get_pooled(self, input_ids, attention_mask,
                   pixel_values, image_grid_thw):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,   # OOM Fix 5
        )
        hidden  = outputs.hidden_states[-1]           # [B, seq_len, hidden]
        seq_len = attention_mask.sum(dim=1) - 1       # last real token
        pooled  = hidden[
            torch.arange(hidden.size(0), device=hidden.device), seq_len
        ]
        return pooled.float()                         # upcast for stability

    def forward(self, input_ids, attention_mask,
                pixel_values, image_grid_thw):
        pooled      = self.get_pooled(input_ids, attention_mask,
                                      pixel_values, image_grid_thw)
        mech_out    = self.mech_head(pooled)
        species_out = (self.species_head(pooled)
                       if self.species_head is not None else None)
        zone_out    = (self.zone_head(pooled)
                       if self.zone_head is not None else None)
        return mech_out, species_out, zone_out


# ══════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════
CLASSIFY_PROMPT = (
    "This is an antibiotic disk diffusion plate. "
    "Analyse the inhibition zones around each disk to determine "
    "bacterial resistance mechanism."
)


class QwenTripleDataset(Dataset):
    def __init__(self, df, processor, zone_norm,
                 sid2row, n_ab, augment=False):
        self.df      = df.reset_index(drop=True)
        self.proc    = processor
        self.zone    = zone_norm
        self.sid2row = sid2row
        self.n_ab    = n_ab
        self.augment = augment

    def __len__(self): return len(self.df)

    def _augment(self, img):
        import torchvision.transforms as T
        return T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.3),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ])(img)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["original_path"]).convert("RGB")
        if self.augment:
            image = self._augment(image)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": CLASSIFY_PROMPT},
            ],
        }]
        text   = self.proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        inputs = self.proc(
            text=[text], images=[image],
            padding="max_length", truncation=True,
            max_length=256,    # OOM Fix 3
            return_tensors="pt",
        )

        r        = self.sid2row.get(row["SampleID"])
        zone_vec = (torch.tensor(self.zone[r], dtype=torch.float32)
                    if r is not None
                    else torch.full((self.n_ab,), float("nan")))

        return {
            "input_ids":      inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values":   inputs["pixel_values"].squeeze(0),
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
            "mech_label":     torch.tensor(int(row["mech_id"]),  dtype=torch.long),
            "spec_label":     torch.tensor(int(row["spec_id"]),  dtype=torch.long),
            "zone_vec":       zone_vec,
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ══════════════════════════════════════════════════════════════════════════
# LOSS UTILITIES
# ══════════════════════════════════════════════════════════════════════════
def make_weights(labels_arr, n_classes):
    counts = Counter(labels_arr.tolist())
    total  = len(labels_arr)
    return torch.tensor(
        [total / (n_classes * counts.get(i, 1)) for i in range(n_classes)],
        dtype=torch.float32).to(DEVICE)


def masked_zone_loss(pred, true):
    mask = ~torch.isnan(true)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return F.smooth_l1_loss(pred[mask], true[mask])


def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


# ══════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL EPOCH
# ══════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, mech_crit, spec_crit,
                model_type, grad_accum=8):
    model.train()
    loss_sum = correct = n = 0
    z_loss_sum = z_count = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        pv   = batch["pixel_values"].to(DEVICE, dtype=DTYPE)
        thw  = batch["image_grid_thw"].to(DEVICE)
        ml   = batch["mech_label"].to(DEVICE)
        sl   = batch["spec_label"].to(DEVICE)
        zv   = batch["zone_vec"].to(DEVICE)

        ml_out, sl_out, zl_out = model(ids, mask, pv, thw)

        loss = mech_crit(ml_out, ml)

        if model_type in ("dual", "triple") and sl_out is not None:
            loss = loss + LAMBDA_SPECIES * spec_crit(sl_out, sl)

        if model_type == "triple" and zl_out is not None:
            zl         = masked_zone_loss(zl_out, zv)
            loss       = loss + LAMBDA_ZONE * zl
            z_loss_sum += zl.item()
            z_count    += 1

        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(
                get_trainable_params(model), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        bs       = ml.size(0)
        loss_sum += loss.item() * bs
        correct  += (ml_out.argmax(1) == ml).sum().item()
        n        += bs

        # Aggressively free VRAM each step
        del ids, mask, pv, thw, ml, sl, zv, ml_out, sl_out, zl_out, loss
        torch.cuda.empty_cache()

    return (loss_sum / n, correct / n,
            z_loss_sum / z_count if z_count > 0 else None)


@torch.no_grad()
def eval_epoch(model, loader, mech_crit, zone_std, model_type):
    model.eval()
    loss_sum = 0
    mt, mp, st, sp = [], [], [], []
    zone_maes = []

    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        pv   = batch["pixel_values"].to(DEVICE, dtype=DTYPE)
        thw  = batch["image_grid_thw"].to(DEVICE)
        ml   = batch["mech_label"].to(DEVICE)
        zv   = batch["zone_vec"].to(DEVICE)

        ml_out, sl_out, zl_out = model(ids, mask, pv, thw)

        loss_sum += mech_crit(ml_out, ml).item() * ml.size(0)
        mt.extend(ml.cpu().numpy())
        mp.extend(ml_out.argmax(1).cpu().numpy())

        if sl_out is not None:
            st.extend(batch["spec_label"].numpy())
            sp.extend(sl_out.argmax(1).cpu().numpy())

        if zl_out is not None:
            z_mask = ~torch.isnan(zv)
            if z_mask.sum() > 0:
                mae_mm = (zl_out[z_mask] - zv[z_mask]).abs().mean().item()
                zone_maes.append(mae_mm * zone_std)

        torch.cuda.empty_cache()

    n        = len(mt)
    mech_acc = accuracy_score(mt, mp)
    mech_f1  = f1_score(mt, mp, average="weighted", zero_division=0)
    spec_acc = accuracy_score(st, sp) if sp else None
    zone_mae = float(np.mean(zone_maes)) if zone_maes else None

    return loss_sum / n, mech_acc, mech_f1, mt, mp, spec_acc, zone_mae


# ══════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION (identical structure to CLIP pipeline)
# ══════════════════════════════════════════════════════════════════════════
def run_cv(backbone, processor, hidden_size, data,
           model_type, n_epochs=15, batch_size=2,
           lr=2e-4, n_folds=5, grad_accum=8):
    """
    5-fold stratified CV.
    Effective batch = batch_size × grad_accum = 2 × 8 = 16.
    """
    label = {
        "single": "Single-Task",
        "dual":   "Dual-Task",
        "triple": "Triple-Task",
    }[model_type]

    img_df    = data["img_df"]
    n_species = data["n_species"]
    zone_norm = data["zone_norm"]
    sid2row   = data["sid2row"]
    n_ab      = len(data["ab_list"])
    zone_std  = data["zone_std"]
    mech_arr  = img_df["mech_id"].values

    print(f"\n{'='*62}")
    print(f"  Qwen2.5-VL-3B {label} — {n_folds}-Fold CV")
    print(f"  batch={batch_size}  grad_accum={grad_accum}  "
          f"effective_batch={batch_size*grad_accum}  epochs={n_epochs}")
    print(f"{'='*62}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics, all_true, all_pred, history = [], [], [], []
    all_zone_maes = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(img_df, mech_arr)):
        print(f"\n  Fold {fold+1}/{n_folds}  "
              f"train={len(tr_idx)}  val={len(val_idx)}")

        tr_ds  = QwenTripleDataset(
            img_df.iloc[tr_idx], processor, zone_norm,
            sid2row, n_ab, augment=True)
        val_ds = QwenTripleDataset(
            img_df.iloc[val_idx], processor, zone_norm,
            sid2row, n_ab, augment=False)

        # num_workers=0: avoids CUDA fork issues
        tr_ld  = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_fn,
                            pin_memory=False)
        val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn,
                            pin_memory=False)

        mech_w    = make_weights(mech_arr[tr_idx], len(MECH_CLASSES))
        spec_w    = make_weights(
            img_df.iloc[tr_idx]["spec_id"].values, n_species)
        mech_crit = nn.CrossEntropyLoss(weight=mech_w)
        spec_crit = nn.CrossEntropyLoss(weight=spec_w)

        # Fresh task heads each fold; LoRA backbone is shared
        model     = QwenMultiTaskModel(
            backbone, hidden_size,
            len(MECH_CLASSES), n_species, n_ab,
            model_type=model_type)
        params    = get_trainable_params(model)
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs)

        best_f1 = 0.0
        best_tr = best_pr = []
        best_zone_mae = best_spec_acc = None

        for epoch in range(n_epochs):
            tr_loss, tr_acc, _ = train_epoch(
                model, tr_ld, optimizer, mech_crit, spec_crit,
                model_type, grad_accum=grad_accum)
            (val_loss, val_acc, val_f1,
             vt, vp, spec_acc, zone_mae) = eval_epoch(
                model, val_ld, mech_crit, zone_std, model_type)
            scheduler.step()

            history.append({
                "fold": fold+1, "epoch": epoch+1, "model": model_type,
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "val_f1": val_f1, "zone_mae_mm": zone_mae,
            })

            if val_f1 > best_f1:
                best_f1       = val_f1
                best_tr       = list(vt)
                best_pr       = list(vp)
                best_zone_mae = zone_mae
                best_spec_acc = spec_acc

            if (epoch + 1) % 3 == 0:
                parts = [f"ep {epoch+1:2d}/{n_epochs}",
                         f"tr={tr_acc:.2f}", f"val={val_acc:.2f}",
                         f"f1={val_f1:.3f}"]
                if zone_mae:
                    parts.append(f"zMAE={zone_mae:.1f}mm")
                print("    " + "  ".join(parts))

            gc.collect()
            torch.cuda.empty_cache()

        all_true.extend([MECH_CLASSES[l] for l in best_tr])
        all_pred.extend([MECH_CLASSES[p] for p in best_pr])
        if best_zone_mae:
            all_zone_maes.append(best_zone_mae)

        fm = {
            "fold":        fold+1,
            "accuracy":    accuracy_score(best_tr, best_pr),
            "weighted_f1": best_f1,
            "kappa":       cohen_kappa_score(best_tr, best_pr),
            "species_acc": best_spec_acc,
            "zone_mae_mm": best_zone_mae,
        }
        fold_metrics.append(fm)

        line = (f"  Fold {fold+1} → acc={fm['accuracy']:.3f}  "
                f"f1={fm['weighted_f1']:.3f}  κ={fm['kappa']:.3f}")
        if best_zone_mae:
            line += f"  zMAE={best_zone_mae:.1f}mm"
        print(line)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    mdf       = pd.DataFrame(fold_metrics)
    spec_vals = [x for x in mdf["species_acc"] if x is not None]
    zmae_vals = [x for x in mdf["zone_mae_mm"] if x is not None]

    print(f"\n── Qwen {label} Summary ─────────────────────────────")
    print(f"  Mech acc   : {mdf['accuracy'].mean()*100:.1f}% "
          f"± {mdf['accuracy'].std()*100:.1f}%")
    print(f"  Weighted F1: {mdf['weighted_f1'].mean():.3f} "
          f"± {mdf['weighted_f1'].std():.3f}")
    print(f"  Kappa      : {mdf['kappa'].mean():.3f} "
          f"± {mdf['kappa'].std():.3f}")
    if zmae_vals:
        print(f"  Zone MAE   : {np.mean(zmae_vals):.1f}mm "
              f"± {np.std(zmae_vals):.1f}mm")
    if spec_vals:
        print(f"  Species acc: {np.mean(spec_vals)*100:.1f}% "
              f"± {np.std(spec_vals)*100:.1f}%")
    print(f"\nMechanism classification report:")
    print(classification_report(
        all_true, all_pred, labels=MECH_CLASSES, zero_division=0))

    mdf.to_csv(OUT_DIR / f"cv_{model_type}.csv", index=False)

    return {
        "model_type":    model_type,
        "mean_acc":      float(mdf["accuracy"].mean()),
        "std_acc":       float(mdf["accuracy"].std()),
        "mean_f1":       float(mdf["weighted_f1"].mean()),
        "std_f1":        float(mdf["weighted_f1"].std()),
        "mean_kappa":    float(mdf["kappa"].mean()),
        "mean_zone_mae": float(np.mean(zmae_vals)) if zmae_vals else None,
        "std_zone_mae":  float(np.std(zmae_vals))  if zmae_vals else None,
        "y_true":  all_true,
        "y_pred":  all_pred,
        "history": pd.DataFrame(history),
        "fold_df": mdf,
    }


# ══════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════
def plot_three_way_comparison(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Qwen2.5-VL-3B: Single / Dual / Triple Task\n5-Fold CV",
        fontsize=13, fontweight="bold")
    keys    = ["mean_acc",  "mean_f1",      "mean_kappa"]
    std_k   = ["std_acc",   "std_f1",       "std_acc"]
    ylabels = ["Accuracy",  "Weighted F1",  "Cohen Kappa"]
    colors  = ["#888780",   "#378ADD",      "#1D9E75"]
    labels  = ["Single-Task", "Dual-Task\n(+Species)",
               "Triple-Task\n(+Species+Zone)"]

    for ax, key, sk, ylabel in zip(axes, keys, std_k, ylabels):
        vals = [results[t][key] for t in ("single", "dual", "triple")]
        errs = [results[t][sk]  for t in ("single", "dual", "triple")]
        bars = ax.bar(labels, vals, yerr=errs, color=colors,
                      alpha=0.85, capsize=6, edgecolor="white", width=0.55)
        ax.set_ylim(0, 1.08)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.012,
                    f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
        delta = results["triple"][key] - results["single"][key]
        sign  = "+" if delta >= 0 else ""
        col   = "#085041" if delta >= 0 else "#A32D2D"
        ax.text(1, max(vals) + 0.06, f"Gain: {sign}{delta:.3f}",
                ha="center", fontsize=9, color=col, fontweight="bold")

    plt.tight_layout()
    path = OUT_DIR / "1_three_way_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


def plot_confusion_trio(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Qwen2.5-VL-3B — Confusion Matrices (5-Fold CV)",
        fontsize=13, fontweight="bold")
    titles = {
        "single": "Single-Task\n(mechanism only)",
        "dual":   "Dual-Task\n(+ species)",
        "triple": "Triple-Task\n(+ species + zone)",
    }
    for mtype, ax in zip(("single", "dual", "triple"), axes):
        res    = results[mtype]
        cm     = confusion_matrix(res["y_true"], res["y_pred"],
                                  labels=MECH_CLASSES)
        cm_n   = cm.astype(float) / np.maximum(
            cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(MECH_CLASSES)))
        ax.set_yticks(range(len(MECH_CLASSES)))
        ax.set_xticklabels(MECH_CLASSES, rotation=30, ha="right")
        ax.set_yticklabels(MECH_CLASSES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(titles[mtype], fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax)
        for i in range(len(MECH_CLASSES)):
            for j in range(len(MECH_CLASSES)):
                col = "white" if cm_n[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm_n[i,j]:.2f}\n({cm[i,j]})",
                        ha="center", va="center",
                        fontsize=8, fontweight="bold", color=col)
    plt.tight_layout()
    path = OUT_DIR / "2_confusion_trio.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


def plot_per_class_f1(results):
    from sklearn.metrics import f1_score as f1_fn
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(
        "Qwen2.5-VL-3B — Per-Class F1: Single → Dual → Triple",
        fontsize=12, fontweight="bold")
    x     = np.arange(len(MECH_CLASSES))
    width = 0.25
    for mtype, off, color, name in zip(
        ("single", "dual", "triple"),
        (-width, 0, width),
        ("#888780", "#378ADD", "#1D9E75"),
        ("Single", "Dual", "Triple"),
    ):
        res  = results[mtype]
        f1s  = f1_fn(res["y_true"], res["y_pred"],
                     average=None, labels=MECH_CLASSES, zero_division=0)
        bars = ax.bar(x + off, f1s, width, label=name,
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(MECH_CLASSES, fontsize=11)
    ax.set_ylabel("F1"); ax.set_ylim(0, 1.12)
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = OUT_DIR / "3_per_class_f1.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


def plot_training_curves(results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Qwen2.5-VL-3B Training Curves — 5-Fold CV",
                 fontsize=13, fontweight="bold")
    for col, mtype in enumerate(("single", "dual", "triple")):
        hist    = results[mtype]["history"]
        ax_loss = axes[0, col]
        ax_acc  = axes[1, col]
        for fold in hist["fold"].unique():
            fh = hist[hist["fold"] == fold]
            ax_loss.plot(fh["epoch"], fh["train_loss"],
                         alpha=0.35, lw=1, color="#378ADD")
            ax_loss.plot(fh["epoch"], fh["val_loss"],
                         alpha=0.35, lw=1, color="#E24B4A")
            ax_acc.plot(fh["epoch"], fh["val_acc"],
                        alpha=0.6, lw=1.5, color="#1D9E75")
        ax_loss.set_title(f"{mtype.capitalize()} — Loss",
                          fontsize=11, fontweight="bold")
        ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
        ax_loss.legend(["Train", "Val"], fontsize=8)
        ax_acc.set_title(f"{mtype.capitalize()} — Val Acc",
                         fontsize=11, fontweight="bold")
        ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1)
    plt.tight_layout()
    path = OUT_DIR / "4_training_curves.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


def plot_qwen_vs_clip(results,
                      clip_acc=0.810, clip_f1=0.789, clip_kappa=0.628):
    best_key = max(("single", "dual", "triple"),
                   key=lambda k: results[k]["mean_f1"])
    best = results[best_key]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(
        f"Best Qwen2.5-VL-3B ({best_key}-task LoRA) vs CLIP Fine-Tuned",
        fontsize=12, fontweight="bold")
    for ax, (name, qv, cv, qs) in zip(axes, [
        ("Accuracy",    best["mean_acc"],   clip_acc,   best["std_acc"]),
        ("Weighted F1", best["mean_f1"],    clip_f1,    best["std_f1"]),
        ("Cohen Kappa", best["mean_kappa"], clip_kappa, 0.0),
    ]):
        bars = ax.bar(
            [f"Qwen LoRA\n({best_key})", "CLIP\n(fine-tuned)"],
            [qv, cv], yerr=[qs, 0],
            color=["#7F77DD", "#1D9E75"], alpha=0.85,
            edgecolor="white", capsize=6, width=0.5)
        ax.set_ylim(0, 1.1)
        ax.set_title(name, fontsize=12, fontweight="bold")
        for bar, val in zip(bars, [qv, cv]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.015,
                    f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
        delta = qv - cv
        sign  = "+" if delta >= 0 else ""
        col   = "#3C3489" if delta >= 0 else "#A32D2D"
        ax.text(0.5, 0.92, f"Δ = {sign}{delta:.3f}",
                transform=ax.transAxes, ha="center", fontsize=10,
                fontweight="bold", color=col,
                bbox=dict(boxstyle="round", facecolor="white",
                          edgecolor=col, alpha=0.8))
    plt.tight_layout()
    path = OUT_DIR / "5_qwen_vs_clip.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


def print_final_table(results):
    print("\n" + "=" * 72)
    print("  FINAL RESULTS — Qwen2.5-VL-3B LoRA Multi-Task (5-Fold CV)")
    print("=" * 72)
    print(f"  {'Model':<30} {'Acc':>8} {'F1':>8} {'Kappa':>8} {'ZoneMAE':>10}")
    print("-" * 72)
    print(f"  {'CLIP fine-tuned (baseline)':<30} "
          f"{'81.0%':>8} {'0.789':>8} {'0.628':>8} {'—':>10}")
    for mtype, label in [
        ("single", "Qwen Single-Task (LoRA)"),
        ("dual",   "Qwen Dual-Task   (LoRA)"),
        ("triple", "Qwen Triple-Task (LoRA)"),
    ]:
        r    = results[mtype]
        zmae = f"{r['mean_zone_mae']:.1f}mm" if r.get("mean_zone_mae") else "—"
        print(f"  {label:<30} "
              f"{r['mean_acc']*100:>7.1f}% "
              f"{r['mean_f1']:>8.3f} "
              f"{r['mean_kappa']:>8.3f} "
              f"{zmae:>10}")
    print("=" * 72)
    gain = results["triple"]["mean_f1"] - results["single"]["mean_f1"]
    print(f"\n  Multi-task F1 gain (single→triple): "
          f"{'+' if gain>=0 else ''}{gain:.3f}")
    if results["triple"].get("mean_zone_mae"):
        print(f"  Zone MAE: {results['triple']['mean_zone_mae']:.1f} ± "
              f"{results['triple']['std_zone_mae']:.1f} mm")
    print("=" * 72)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def run_pipeline():
    data = load_data()

    backbone, processor = load_qwen_backbone()

    try:
        hidden_size = backbone.config.hidden_size
    except AttributeError:
        hidden_size = 2048
    print(f"LM hidden size: {hidden_size}")

    results = {}

    for mtype in ("single", "dual", "triple"):
        results[mtype] = run_cv(
            backbone, processor, hidden_size, data,
            model_type=mtype,
            n_epochs=15,
            batch_size=2,      # OOM Fix 7
            lr=2e-4,
            n_folds=5,
            grad_accum=8,      # effective batch = 16
        )

    plot_three_way_comparison(results)
    plot_confusion_trio(results)
    plot_per_class_f1(results)
    plot_training_curves(results)
    plot_qwen_vs_clip(results)
    print_final_table(results)

    pd.concat([r["history"] for r in results.values()]).to_csv(
        OUT_DIR / "all_histories.csv", index=False)
    print(f"\nAll outputs → {OUT_DIR}")
    return results


if __name__ == "__main__":
    results = run_pipeline()
