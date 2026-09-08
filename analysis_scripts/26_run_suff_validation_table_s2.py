#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run full BTCV validation with and without SUFF for Table S2.

Outputs:
    1. Per-case metrics:
       /data/oldproject/analysis_outputs/table_s2_suff_validation/table_s2_case_metrics.csv

    2. Summary table:
       /data/oldproject/analysis_outputs/table_s2_suff_validation/table_s2_summary.csv
       /data/oldproject/analysis_outputs/table_s2_suff_validation/table_s2_summary.txt

Metrics:
    - Mean Dice: mean over 13 foreground organ channels
    - HD95: MONAI HausdorffDistanceMetric, percentile=95
    - FP voxel ratio: FP / (TP + FP) * 100, foreground-merged
    - FN voxel ratio: FN / (TP + FN) * 100, foreground-merged

Important:
    SUFF is implemented as an inference-stage post-processing / fusion module
    based on DDIM intermediate outputs.
"""

import os
import sys
import csv
import time
import random
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

# =========================
# Settings
# =========================

PROJECT_ROOT = Path("/data/oldproject/exp/Diff-UNet-main2")
BTCV_ROOT = PROJECT_ROOT / "BTCV"
DATA_DIR = str(BTCV_ROOT / "RawData/Training") + "/"

CKPT_PATH = BTCV_ROOT / "logs_btcv" / "best_model_0.7680.pt"

ROI_SIZE = [96, 96, 96]
SW_OVERLAP = 0.5

NOISE_SAMPLES = 5
BASE_SEED = 20260825

D_LOW = 0.05
D_HIGH = 0.45
BUTTERWORTH_ORDER = 2

# Conservative SUFF v2 settings.
LOW_FREQ_WEIGHT = 1.00
BAND_FREQ_WEIGHT = 1.05
HIGH_FREQ_WEIGHT = 0.70
ANCHOR_ALPHA = 0.85

EPS = 1e-6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = Path("/data/oldproject/analysis_outputs/table_s2_suff_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASE_CSV = OUT_DIR / "table_s2_case_metrics.csv"
SUMMARY_CSV = OUT_DIR / "table_s2_summary.csv"
SUMMARY_TXT = OUT_DIR / "table_s2_summary.txt"

RESUME = True

# =========================
# Imports from original project
# =========================

sys.path.insert(0, str(BTCV_ROOT))

from train import DiffUNet
import train as train_module

from monai.inferers import SlidingWindowInferer
from monai.metrics import HausdorffDistanceMetric


# =========================
# Utility
# =========================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_state_dict_flexible(model, ckpt_path: Path):
    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    clean_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        clean_state[nk] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)

    print(f"missing keys: {len(missing)}", flush=True)
    print(f"unexpected keys: {len(unexpected)}", flush=True)

    if len(missing) < 20 and len(unexpected) < 20:
        if missing:
            print("missing:", missing, flush=True)
        if unexpected:
            print("unexpected:", unexpected, flush=True)

    return model


def get_btcv_val_dataset():
    if not hasattr(train_module, "get_loader_btcv"):
        raise RuntimeError("train.py does not expose get_loader_btcv.")

    fn = train_module.get_loader_btcv

    try:
        loaded = fn(data_dir=DATA_DIR)
    except TypeError:
        try:
            loaded = fn(DATA_DIR)
        except TypeError:
            loaded = fn()

    if isinstance(loaded, (list, tuple)):
        if len(loaded) >= 2:
            val_obj = loaded[1]
        else:
            raise RuntimeError("get_loader_btcv returned list/tuple with length < 2.")
    elif isinstance(loaded, dict):
        for key in ["val", "valid", "validation", "val_ds", "val_loader"]:
            if key in loaded:
                val_obj = loaded[key]
                break
        else:
            raise RuntimeError(f"Could not find validation object in keys: {list(loaded.keys())}")
    else:
        raise RuntimeError(f"Unsupported get_loader_btcv return type: {type(loaded)}")

    if hasattr(val_obj, "dataset"):
        return val_obj.dataset
    return val_obj


def prepare_case_batch(val_ds, case_idx: int):
    item = val_ds[case_idx]

    image = item["image"]
    label = item["label"]

    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    if not torch.is_tensor(label):
        label = torch.as_tensor(label)

    if image.ndim == 4:
        image = image.unsqueeze(0)
    if label.ndim == 4:
        label = label.unsqueeze(0)

    image = image.float()
    label = label.long()

    return image, label, item


def raw_label_to_13ch(label_b1dhw: torch.Tensor):
    """
    Convert raw label map [B,1,D,H,W] to [B,13,D,H,W].
    This matches train.py convert_labels(): labels == 1 ... 13.
    """
    if label_b1dhw.ndim != 5:
        raise ValueError(f"label should be [B,1,D,H,W], got {tuple(label_b1dhw.shape)}")

    label = label_b1dhw[:, 0]
    channels = []
    for cls_id in range(1, 14):
        channels.append((label == cls_id).float())
    return torch.stack(channels, dim=1)


def prob_to_logit(p, eps=1e-6):
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def make_3d_butterworth_weight(shape_dhw, device, dtype=torch.float32):
    d, h, w = shape_dhw

    fz = torch.fft.fftfreq(d, device=device, dtype=dtype)
    fy = torch.fft.fftfreq(h, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(w, device=device, dtype=dtype)

    zz, yy, xx = torch.meshgrid(fz, fy, fx, indexing="ij")
    r = torch.sqrt(zz ** 2 + yy ** 2 + xx ** 2)
    r = r / (r.max() + EPS)

    order = BUTTERWORTH_ORDER

    low_soft = 1.0 / (1.0 + (r / (D_LOW + EPS)) ** (2 * order))
    high_soft = 1.0 - 1.0 / (1.0 + (r / (D_HIGH + EPS)) ** (2 * order))

    lowpass_high = 1.0 / (1.0 + (r / (D_HIGH + EPS)) ** (2 * order))
    lowpass_low = 1.0 / (1.0 + (r / (D_LOW + EPS)) ** (2 * order))
    band_soft = lowpass_high * (1.0 - lowpass_low)

    part_sum = low_soft + band_soft + high_soft + EPS
    low_soft = low_soft / part_sum
    band_soft = band_soft / part_sum
    high_soft = high_soft / part_sum

    weight = (
        LOW_FREQ_WEIGHT * low_soft
        + BAND_FREQ_WEIGHT * band_soft
        + HIGH_FREQ_WEIGHT * high_soft
    )

    return weight


def suff_fuse_from_all_probs(all_probs_cpu: torch.Tensor, suff_device: str):
    """
    all_probs_cpu:
        [N, T, C, D, H, W], probability maps on CPU.

    Return:
        [1, C, D, H, W] probability map on CPU.
    """
    if all_probs_cpu.ndim != 6:
        raise ValueError(f"Expected [N,T,C,D,H,W], got {tuple(all_probs_cpu.shape)}")

    n, t, c, d, h, w = all_probs_cpu.shape

    freq_weight = make_3d_butterworth_weight(
        (d, h, w),
        device=suff_device,
        dtype=torch.float32,
    )

    fused_classes = []

    with torch.no_grad():
        for cls in range(c):
            x = all_probs_cpu[:, :, cls].to(suff_device, dtype=torch.float32)
            x_fft = torch.fft.fftn(x, dim=(-3, -2, -1))

            x_fft = x_fft * freq_weight[None, None, :, :, :]

            mag = torch.abs(x_fft)
            var_nt = torch.var(mag, dim=0, unbiased=False)

            inv_uncertainty = 1.0 / (var_nt + EPS)
            inv_uncertainty = torch.clamp(inv_uncertainty, max=1e6)

            w_t = inv_uncertainty / (inv_uncertainty.sum(dim=0, keepdim=True) + EPS)

            mean_fft_t = x_fft.mean(dim=0)
            fused_fft = (w_t * mean_fft_t).sum(dim=0)

            fused = torch.fft.ifftn(fused_fft, dim=(-3, -2, -1)).real
            fused = torch.clamp(fused, 0.0, 1.0)

            fused_classes.append(fused.cpu())

            del x, x_fft, mag, var_nt, inv_uncertainty, w_t, mean_fft_t, fused_fft, fused
            if suff_device.startswith("cuda"):
                torch.cuda.empty_cache()

    fused_prob = torch.stack(fused_classes, dim=0).unsqueeze(0)
    return fused_prob


class FinalAndSuffPredictor(torch.nn.Module):
    """
    SlidingWindowInferer-compatible wrapper.

    Return:
        [B, 26, D, H, W]
        channels 0:13  = final DDIM raw output
        channels 13:26 = SUFF raw-like output
    """

    def __init__(self, diffunet, noise_samples=5, base_seed=20260825):
        super().__init__()
        self.diffunet = diffunet
        self.noise_samples = noise_samples
        self.base_seed = base_seed
        self.call_index = 0

    def forward(self, image):
        device = image.device
        self.diffunet.eval()

        with torch.no_grad():
            embeddings = self.diffunet.embed_model(image)

            all_probs = []
            baseline_raw = None
            final_probs_for_anchor = []

            for s in range(self.noise_samples):
                local_seed = self.base_seed + self.call_index * 1000 + s
                torch.manual_seed(local_seed)
                torch.cuda.manual_seed_all(local_seed)

                sample_out = self.diffunet.sample_diffusion.ddim_sample_loop(
                    self.diffunet.model,
                    (1, 13, ROI_SIZE[0], ROI_SIZE[1], ROI_SIZE[2]),
                    model_kwargs={"image": image, "embeddings": embeddings},
                    progress=False,
                )

                if s == 0:
                    baseline_raw = sample_out["pred_xstart"].detach().cpu()

                final_probs_for_anchor.append(
                    torch.sigmoid(sample_out["pred_xstart"].detach().cpu())
                )

                steps = sample_out["all_samples"]
                steps_tensor = torch.stack([x.squeeze(0).float() for x in steps], dim=0)
                steps_prob = torch.sigmoid(steps_tensor)
                all_probs.append(steps_prob.cpu())

                del sample_out, steps, steps_tensor, steps_prob
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            all_probs = torch.stack(all_probs, dim=0)
            suff_prob_raw = suff_fuse_from_all_probs(all_probs, suff_device=DEVICE)

            mean_final_prob = torch.stack(final_probs_for_anchor, dim=0).mean(dim=0)
            suff_prob = ANCHOR_ALPHA * mean_final_prob + (1.0 - ANCHOR_ALPHA) * suff_prob_raw
            suff_raw = prob_to_logit(suff_prob)

            out = torch.cat([baseline_raw, suff_raw], dim=1).to(device)

            self.call_index += 1

            del all_probs, baseline_raw, final_probs_for_anchor, suff_prob_raw, mean_final_prob, suff_prob, suff_raw
            if device.type == "cuda":
                torch.cuda.empty_cache()

            return out


def dice_per_class(pred, target):
    pred = pred.float()
    target = target.float()

    dims = (0, 2, 3, 4)
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)

    dice = (2.0 * inter + 1e-5) / (denom + 1e-5)
    return dice


def compute_dice_hd95(pred_bin, target_bin):
    """
    pred_bin, target_bin: [1,13,D,H,W]
    """
    dice_cls = dice_per_class(pred_bin, target_bin)
    mean_dice = float(dice_cls.mean().item())

    hd95_value = np.nan
    try:
        hd_metric = HausdorffDistanceMetric(
            include_background=True,
            percentile=95,
            reduction="mean",
        )
        hd = hd_metric(pred_bin.cpu(), target_bin.cpu())
        hd95_value = float(hd.item())
    except Exception as e:
        print("Warning: HD95 failed:", repr(e), flush=True)

    return mean_dice, hd95_value


def compute_fp_fn_voxel_ratios(pred_bin, target_bin):
    """
    Foreground-merged voxel-level FP/FN ratios.

    FP voxel ratio = FP / (TP + FP) * 100
    FN voxel ratio = FN / (TP + FN) * 100
    """
    pred_fg = pred_bin[0].bool().any(dim=0)
    target_fg = target_bin[0].bool().any(dim=0)

    tp = torch.logical_and(pred_fg, target_fg).sum().item()
    fp = torch.logical_and(pred_fg, torch.logical_not(target_fg)).sum().item()
    fn = torch.logical_and(torch.logical_not(pred_fg), target_fg).sum().item()
    tn = torch.logical_and(torch.logical_not(pred_fg), torch.logical_not(target_fg)).sum().item()

    fp_ratio = 100.0 * fp / (tp + fp + EPS)
    fn_ratio = 100.0 * fn / (tp + fn + EPS)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "fp_voxel_ratio_percent": float(fp_ratio),
        "fn_voxel_ratio_percent": float(fn_ratio),
    }


def get_case_name(val_ds, case_idx):
    try:
        if hasattr(val_ds, "datalist"):
            item = val_ds.datalist[case_idx]
            if isinstance(item, dict):
                image_path = str(item.get("image", ""))
                label_path = str(item.get("label", ""))
                return image_path, label_path
    except Exception:
        pass
    return "", ""


def existing_done_cases():
    done = set()
    if RESUME and CASE_CSV.exists():
        with CASE_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    done.add(int(row["case_idx"]))
                except Exception:
                    pass
    return done


def init_case_csv():
    if CASE_CSV.exists() and RESUME:
        return

    fieldnames = [
        "case_idx",
        "image_path",
        "label_path",
        "elapsed_sec",
        "final_ddim_dice",
        "suff_dice",
        "final_ddim_hd95",
        "suff_hd95",
        "final_ddim_fp_voxel_ratio_percent",
        "suff_fp_voxel_ratio_percent",
        "final_ddim_fn_voxel_ratio_percent",
        "suff_fn_voxel_ratio_percent",
        "final_ddim_tp",
        "final_ddim_fp",
        "final_ddim_fn",
        "final_ddim_tn",
        "suff_tp",
        "suff_fp",
        "suff_fn",
        "suff_tn",
    ]

    with CASE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_case_row(row):
    fieldnames = [
        "case_idx",
        "image_path",
        "label_path",
        "elapsed_sec",
        "final_ddim_dice",
        "suff_dice",
        "final_ddim_hd95",
        "suff_hd95",
        "final_ddim_fp_voxel_ratio_percent",
        "suff_fp_voxel_ratio_percent",
        "final_ddim_fn_voxel_ratio_percent",
        "suff_fn_voxel_ratio_percent",
        "final_ddim_tp",
        "final_ddim_fp",
        "final_ddim_fn",
        "final_ddim_tn",
        "suff_tp",
        "suff_fp",
        "suff_fn",
        "suff_tn",
    ]

    with CASE_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def build_summary():
    rows = []
    with CASE_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print("No rows found in case CSV.", flush=True)
        return

    methods = {
        "Final DDIM output": {
            "dice": [float(r["final_ddim_dice"]) for r in rows],
            "hd95": [float(r["final_ddim_hd95"]) for r in rows],
            "fp_ratio": [float(r["final_ddim_fp_voxel_ratio_percent"]) for r in rows],
            "fn_ratio": [float(r["final_ddim_fn_voxel_ratio_percent"]) for r in rows],
        },
        "SUFF-processed output": {
            "dice": [float(r["suff_dice"]) for r in rows],
            "hd95": [float(r["suff_hd95"]) for r in rows],
            "fp_ratio": [float(r["suff_fp_voxel_ratio_percent"]) for r in rows],
            "fn_ratio": [float(r["suff_fn_voxel_ratio_percent"]) for r in rows],
        },
    }

    summary_rows = []
    for method, vals in methods.items():
        dice_m, dice_s = mean_std(vals["dice"])
        hd_m, hd_s = mean_std(vals["hd95"])
        fp_m, fp_s = mean_std(vals["fp_ratio"])
        fn_m, fn_s = mean_std(vals["fn_ratio"])

        summary_rows.append({
            "Method": method,
            "Mean Dice": f"{dice_m:.4f} ± {dice_s:.4f}",
            "HD95 (mm)": f"{hd_m:.4f} ± {hd_s:.4f}",
            "FP voxel ratio (%)": f"{fp_m:.4f} ± {fp_s:.4f}",
            "FN voxel ratio (%)": f"{fn_m:.4f} ± {fn_s:.4f}",
        })

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Method",
                "Mean Dice",
                "HD95 (mm)",
                "FP voxel ratio (%)",
                "FN voxel ratio (%)",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write("Table S2. Quantitative comparison of segmentation performance with and without SUFF on the BTCV validation set.\n")
        f.write("=" * 100 + "\n")
        f.write("FP voxel ratio = FP / (TP + FP) * 100\n")
        f.write("FN voxel ratio = FN / (TP + FN) * 100\n\n")
        for r in summary_rows:
            f.write(
                f"{r['Method']}: "
                f"Dice {r['Mean Dice']}, "
                f"HD95 {r['HD95 (mm)']}, "
                f"FP voxel ratio {r['FP voxel ratio (%)']}, "
                f"FN voxel ratio {r['FN voxel ratio (%)']}\n"
            )

    print("\n" + "=" * 80, flush=True)
    print("Summary saved:", SUMMARY_CSV, flush=True)
    print("Summary saved:", SUMMARY_TXT, flush=True)
    print("=" * 80, flush=True)
    for r in summary_rows:
        print(r, flush=True)


def main():
    print("=" * 80, flush=True)
    print("Run full BTCV validation for Table S2: Final DDIM vs SUFF", flush=True)
    print("=" * 80, flush=True)
    print("PROJECT_ROOT:", PROJECT_ROOT, flush=True)
    print("DATA_DIR:", DATA_DIR, flush=True)
    print("CKPT_PATH:", CKPT_PATH, flush=True)
    print("DEVICE:", DEVICE, flush=True)
    print("NOISE_SAMPLES:", NOISE_SAMPLES, flush=True)
    print("ROI_SIZE:", ROI_SIZE, flush=True)
    print("SW_OVERLAP:", SW_OVERLAP, flush=True)
    print("ANCHOR_ALPHA:", ANCHOR_ALPHA, flush=True)
    print("=" * 80, flush=True)

    seed_everything(BASE_SEED)

    val_ds = get_btcv_val_dataset()
    print("val dataset length:", len(val_ds), flush=True)

    model = DiffUNet().to(DEVICE)
    model = load_state_dict_flexible(model, CKPT_PATH)
    model.eval()

    predictor = FinalAndSuffPredictor(
        model,
        noise_samples=NOISE_SAMPLES,
        base_seed=BASE_SEED,
    ).to(DEVICE)
    predictor.eval()

    inferer = SlidingWindowInferer(
        roi_size=ROI_SIZE,
        sw_batch_size=1,
        overlap=SW_OVERLAP,
    )

    init_case_csv()
    done_cases = existing_done_cases()

    total_start = time.time()

    for case_idx in range(len(val_ds)):
        if case_idx in done_cases:
            print(f"[SKIP] case {case_idx} already exists in CSV.", flush=True)
            continue

        print("\n" + "=" * 80, flush=True)
        print(f"Running case {case_idx + 1}/{len(val_ds)} | case_idx={case_idx}", flush=True)
        print("=" * 80, flush=True)

        image_path, label_path = get_case_name(val_ds, case_idx)
        if image_path:
            print("image:", image_path, flush=True)
            print("label:", label_path, flush=True)

        image, label, item = prepare_case_batch(val_ds, case_idx)
        target_13 = raw_label_to_13ch(label)

        print("image shape:", tuple(image.shape), flush=True)
        print("label shape:", tuple(label.shape), flush=True)

        case_start = time.time()

        with torch.no_grad():
            out_raw = inferer(image.to(DEVICE), predictor)

        elapsed = time.time() - case_start

        out_raw = out_raw.detach().cpu()
        base_prob = torch.sigmoid(out_raw[:, :13])
        suff_prob = torch.sigmoid(out_raw[:, 13:26])

        base_bin = (base_prob > 0.5).float()
        suff_bin = (suff_prob > 0.5).float()

        base_dice, base_hd95 = compute_dice_hd95(base_bin, target_13)
        suff_dice, suff_hd95 = compute_dice_hd95(suff_bin, target_13)

        base_fpfn = compute_fp_fn_voxel_ratios(base_bin, target_13)
        suff_fpfn = compute_fp_fn_voxel_ratios(suff_bin, target_13)

        print(f"elapsed_sec: {elapsed:.2f}", flush=True)
        print(f"Final DDIM Dice: {base_dice:.6f} | HD95: {base_hd95}", flush=True)
        print(f"SUFF Dice:       {suff_dice:.6f} | HD95: {suff_hd95}", flush=True)
        print(f"Final DDIM FP voxel ratio: {base_fpfn['fp_voxel_ratio_percent']:.4f}% | FN voxel ratio: {base_fpfn['fn_voxel_ratio_percent']:.4f}%", flush=True)
        print(f"SUFF FP voxel ratio:       {suff_fpfn['fp_voxel_ratio_percent']:.4f}% | FN voxel ratio: {suff_fpfn['fn_voxel_ratio_percent']:.4f}%", flush=True)

        row = {
            "case_idx": case_idx,
            "image_path": image_path,
            "label_path": label_path,
            "elapsed_sec": f"{elapsed:.2f}",
            "final_ddim_dice": f"{base_dice:.6f}",
            "suff_dice": f"{suff_dice:.6f}",
            "final_ddim_hd95": f"{base_hd95:.6f}" if not np.isnan(base_hd95) else "nan",
            "suff_hd95": f"{suff_hd95:.6f}" if not np.isnan(suff_hd95) else "nan",
            "final_ddim_fp_voxel_ratio_percent": f"{base_fpfn['fp_voxel_ratio_percent']:.6f}",
            "suff_fp_voxel_ratio_percent": f"{suff_fpfn['fp_voxel_ratio_percent']:.6f}",
            "final_ddim_fn_voxel_ratio_percent": f"{base_fpfn['fn_voxel_ratio_percent']:.6f}",
            "suff_fn_voxel_ratio_percent": f"{suff_fpfn['fn_voxel_ratio_percent']:.6f}",
            "final_ddim_tp": base_fpfn["tp"],
            "final_ddim_fp": base_fpfn["fp"],
            "final_ddim_fn": base_fpfn["fn"],
            "final_ddim_tn": base_fpfn["tn"],
            "suff_tp": suff_fpfn["tp"],
            "suff_fp": suff_fpfn["fp"],
            "suff_fn": suff_fpfn["fn"],
            "suff_tn": suff_fpfn["tn"],
        }

        append_case_row(row)
        build_summary()

        del image, label, target_13, out_raw, base_prob, suff_prob, base_bin, suff_bin
        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80, flush=True)
    print(f"All done. Total elapsed sec: {total_elapsed:.2f}", flush=True)
    print("=" * 80, flush=True)

    build_summary()


if __name__ == "__main__":
    main()