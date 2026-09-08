#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Case-level SUFF post-processing for BTCV Diff-UNet / MAFF-DiffSeg.

Purpose:
    Compare conventional final DDIM prediction with SUFF-processed prediction.

Logic:
    before SUFF:
        final DDIM pred_xstart from the last denoising iteration.

    after SUFF:
        all DDIM intermediate pred_xstart outputs
        -> sigmoid probability maps
        -> 3D FFT
        -> 3D Butterworth frequency weighting
        -> uncertainty-aware fusion across DDIM timesteps
        -> inverse FFT
        -> final SUFF probability map.

This script does not change model training.
It implements SUFF as an inference-stage post-processing / fusion module.
"""

import os
import sys
import time
import random
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# User settings
# =========================

PROJECT_ROOT = Path("/data/oldproject/exp/Diff-UNet-main2")
BTCV_ROOT = PROJECT_ROOT / "BTCV"

DATA_DIR = str(BTCV_ROOT / "RawData/Training") + "/"

CKPT_PATH = BTCV_ROOT / "logs_btcv" / "best_model_0.7680.pt"

CASE_INDEX = 9

ROI_SIZE = [96, 96, 96]
SW_OVERLAP = 0.5

DDIM_STEPS = 10
NOISE_SAMPLES = 5

BASE_SEED = 20260825

D_LOW = 0.05
D_HIGH = 0.45
BUTTERWORTH_ORDER = 2

# Frequency-region weights.
# The band-pass region is kept strongest.
LOW_FREQ_WEIGHT = 1.00
BAND_FREQ_WEIGHT = 1.05
HIGH_FREQ_WEIGHT = 0.70

ANCHOR_ALPHA = 0.85

EPS = 1e-6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = Path("/data/oldproject/analysis_outputs/suff_case")
FIG_DIR = Path("/data/oldproject/analysis_outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Project imports
# =========================

sys.path.insert(0, str(BTCV_ROOT))

try:
    from train import DiffUNet
except Exception as e:
    raise RuntimeError(
        "Could not import DiffUNet from BTCV/train.py. "
        "Please run this script from the original environment."
    ) from e

try:
    from monai.inferers import SlidingWindowInferer
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
except Exception as e:
    raise RuntimeError(
        "Could not import MONAI. Please activate /data/conda_envs/diffseg first."
    ) from e

try:
    import train as train_module
except Exception as e:
    raise RuntimeError("Could not import BTCV/train.py as train_module.") from e


# =========================
# Utility functions
# =========================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_state_dict_flexible(model, ckpt_path: Path):
    print(f"Loading checkpoint: {ckpt_path}")
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
        if nk.startswith("model."):
            # Do not strip blindly here. DiffUNet itself has submodule named "model".
            # Keep original key.
            nk = k
        clean_state[nk] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)

    print(f"missing keys: {len(missing)}")
    print(f"unexpected keys: {len(unexpected)}")

    if len(missing) < 20 and len(unexpected) < 20:
        if missing:
            print("missing:", missing)
        if unexpected:
            print("unexpected:", unexpected)

    return model


def get_btcv_val_dataset():
    """
    Use the same get_loader_btcv function that train.py imported.
    This is intentionally flexible because different copies of the project
    may return datasets, loaders, tuples, or dicts.
    """
    if not hasattr(train_module, "get_loader_btcv"):
        raise RuntimeError(
            "train.py does not expose get_loader_btcv. "
            "Please check the data loader import in BTCV/train.py."
        )

    fn = train_module.get_loader_btcv

    try:
        loaded = fn(data_dir=DATA_DIR)
    except TypeError:
        try:
            loaded = fn(DATA_DIR)
        except TypeError:
            loaded = fn()

    # Common case: tuple/list = train, val, test
    if isinstance(loaded, (list, tuple)):
        if len(loaded) >= 2:
            val_obj = loaded[1]
        else:
            raise RuntimeError("get_loader_btcv returned a list/tuple with length < 2.")
    elif isinstance(loaded, dict):
        for key in ["val", "valid", "validation", "val_ds", "val_loader"]:
            if key in loaded:
                val_obj = loaded[key]
                break
        else:
            raise RuntimeError(f"Could not find validation dataset/loader in keys: {list(loaded.keys())}")
    else:
        raise RuntimeError(f"Unsupported get_loader_btcv return type: {type(loaded)}")

    # If it is a DataLoader, get its dataset.
    if hasattr(val_obj, "dataset"):
        val_ds = val_obj.dataset
    else:
        val_ds = val_obj

    return val_ds


def prepare_case_batch(val_ds, case_idx: int):
    item = val_ds[case_idx]

    image = item["image"]
    label = item["label"]

    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    if not torch.is_tensor(label):
        label = torch.as_tensor(label)

    # Expected image: [1, D, H, W]
    # Expected label: [1, D, H, W]
    if image.ndim == 4:
        image = image.unsqueeze(0)
    if label.ndim == 4:
        label = label.unsqueeze(0)

    image = image.float()
    label = label.long()

    return image, label, item


def raw_label_to_13ch(label_b1dhw: torch.Tensor):
    """
    Convert BTCV raw label map [1, 1, D, H, W] to one-hot foreground labels
    [1, 13, D, H, W].
    """
    if label_b1dhw.ndim != 5:
        raise ValueError(f"label should be [B,1,D,H,W], got {tuple(label_b1dhw.shape)}")

    label = label_b1dhw[:, 0]
    channels = []
    for cls_id in range(1, 14):
        channels.append((label == cls_id).float())
    out = torch.stack(channels, dim=1)
    return out


def normalize_ct(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return np.zeros_like(x)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo)

def prob_to_logit(p, eps=1e-6):
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))

def make_3d_butterworth_weight(shape_dhw, device, dtype=torch.float32):
    """
    Create a smooth 3D frequency weighting map.

    Frequency radius is normalized to [0, 1] using the maximum 3D radius.
    The band-pass region receives the highest weight.
    """
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

    # Butterworth-like band-pass:
    lowpass_high = 1.0 / (1.0 + (r / (D_HIGH + EPS)) ** (2 * order))
    lowpass_low = 1.0 / (1.0 + (r / (D_LOW + EPS)) ** (2 * order))
    band_soft = lowpass_high * (1.0 - lowpass_low)

    # Normalize soft partitions so their sum is stable.
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
        [N, T, C, D, H, W], probability maps in CPU float32.

    Returns:
        fused probability map [1, C, D, H, W] on CPU.
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
            # [N, T, D, H, W]
            x = all_probs_cpu[:, :, cls].to(suff_device, dtype=torch.float32)

            # Frequency-domain transform.
            # [N, T, D, H, W], complex64
            x_fft = torch.fft.fftn(x, dim=(-3, -2, -1))

            # Apply frequency weighting.
            x_fft = x_fft * freq_weight[None, None, :, :, :]

            # Uncertainty: variance across noise samples at each timestep/frequency.
            # [T, D, H, W]
            mag = torch.abs(x_fft)
            var_nt = torch.var(mag, dim=0, unbiased=False)

            inv_uncertainty = 1.0 / (var_nt + EPS)
            inv_uncertainty = torch.clamp(inv_uncertainty, max=1e6)

            # Normalize weights across timesteps.
            # [T, D, H, W]
            w_t = inv_uncertainty / (inv_uncertainty.sum(dim=0, keepdim=True) + EPS)

            # Mean over noise samples for each timestep.
            # [T, D, H, W]
            mean_fft_t = x_fft.mean(dim=0)

            # Weighted fusion across timesteps.
            # [D, H, W]
            fused_fft = (w_t * mean_fft_t).sum(dim=0)

            # Back to image domain.
            fused = torch.fft.ifftn(fused_fft, dim=(-3, -2, -1)).real

            # Probability-like range.
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

    For each ROI patch:
        - run DDIM NOISE_SAMPLES times
        - baseline = final pred_xstart from the first noise realization
        - SUFF = frequency-domain uncertainty-aware fusion of all intermediate outputs

    Return:
        [B, 26, D, H, W]
        channels 0:13  = baseline probability
        channels 13:26 = SUFF probability
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
            baseline_prob = None
            final_probs_for_anchor = []

            for s in range(self.noise_samples):
                # Make each noise realization reproducible but different.
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

                final_probs_for_anchor.append(torch.sigmoid(sample_out["pred_xstart"].detach().cpu()))

                steps = sample_out["all_samples"]
                # steps list length = DDIM steps.
                # each element: [1, 13, D, H, W], CPU tensor from gaussian_diffusion.py
                steps_tensor = torch.stack([x.squeeze(0).float() for x in steps], dim=0)
                steps_prob = torch.sigmoid(steps_tensor)
                all_probs.append(steps_prob.cpu())

                del sample_out, steps, steps_tensor, steps_prob
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            all_probs = torch.stack(all_probs, dim=0)
            # [N, T, C, D, H, W]

            suff_prob_raw = suff_fuse_from_all_probs(all_probs, suff_device=DEVICE)
            
            mean_final_prob = torch.stack(final_probs_for_anchor, dim=0).mean(dim=0)
            
            suff_prob = ANCHOR_ALPHA * mean_final_prob + (1.0 - ANCHOR_ALPHA) * suff_prob_raw
            suff_raw = prob_to_logit(suff_prob)

            out = torch.cat([baseline_raw, suff_raw], dim=1).to(device)

            self.call_index += 1

            del all_probs, baseline_prob, suff_prob
            if device.type == "cuda":
                torch.cuda.empty_cache()

            return out


def dice_per_class(pred, target):
    """
    pred, target: [1, C, D, H, W], binary tensors.
    """
    pred = pred.float()
    target = target.float()
    dims = (0, 2, 3, 4)

    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)

    dice = (2.0 * inter + 1e-5) / (denom + 1e-5)
    return dice


def compute_metrics(pred_bin, target_bin):
    """
    pred_bin, target_bin: [1, 13, D, H, W]
    """
    dice_cls = dice_per_class(pred_bin, target_bin)
    mean_dice = dice_cls.mean().item()

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
        print("Warning: HD95 failed:", repr(e))

    return mean_dice, hd95_value, dice_cls.detach().cpu().numpy()


def foreground_from_prob(prob_bcdhw):
    """
    prob_bcdhw: [1, 13, D, H, W]
    Return:
        fg_prob [D,H,W], fg_mask [D,H,W]
    """
    prob = prob_bcdhw[0].detach().cpu()
    fg_prob = prob.max(dim=0).values.numpy()
    fg_mask = (prob > 0.5).any(dim=0).numpy()
    return fg_prob, fg_mask


def select_s3_slice(image_3d, gt_fg, base_fg, suff_fg):
    removed_fp = base_fg & (~suff_fg) & (~gt_fg)
    added_fp = suff_fg & (~base_fg) & (~gt_fg)
    diff = base_fg != suff_fg

    scores = []
    for z in range(image_3d.shape[0]):
        score = (
            int(removed_fp[z].sum()) * 3
            + int(diff[z].sum())
            + int(gt_fg[z].sum()) * 0.001
            - int(added_fp[z].sum()) * 0.5
        )
        scores.append(score)

    z_best = int(np.argmax(scores))
    return z_best, removed_fp, added_fp, diff


def plot_s3(image_3d, gt_fg, base_fg, suff_fg, z, case_idx):
    ct = normalize_ct(image_3d[z])

    removed_fp = base_fg & (~suff_fg) & (~gt_fg)
    added_fp = suff_fg & (~base_fg) & (~gt_fg)
    diff = base_fg != suff_fg

    panels = [
        ("Input CT", None),
        ("Ground truth", gt_fg[z]),
        ("Final DDIM pred", base_fg[z]),
        ("SUFF pred", suff_fg[z]),
        ("Removed FP", removed_fp[z]),
        ("Added FP", added_fp[z]),
        ("Prediction difference", diff[z]),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(18, 3.4))

    for ax, (title, mask) in zip(axes, panels):
        ax.imshow(ct, cmap="gray")
        if mask is not None:
            masked = np.ma.masked_where(mask <= 0, mask)
            ax.imshow(masked, alpha=0.55)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(f"Case {case_idx:03d}, z={z}: final DDIM vs SUFF", fontsize=11)
    plt.tight_layout()

    out_png = FIG_DIR / f"Fig_S3_SUFF_case{case_idx:03d}_z{z:03d}.png"
    out_pdf = FIG_DIR / f"Fig_S3_SUFF_case{case_idx:03d}_z{z:03d}.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved S3:", out_png)
    print("Saved S3:", out_pdf)


def fft_log_spectrum_2d(x2d):
    x = np.asarray(x2d, dtype=np.float32)
    x = x - x.mean()
    spec = np.fft.fftshift(np.fft.fft2(x))
    mag = np.log1p(np.abs(spec))
    lo, hi = np.percentile(mag, [1, 99])
    if hi <= lo:
        return np.zeros_like(mag)
    mag = np.clip(mag, lo, hi)
    return (mag - lo) / (hi - lo)


def high_frequency_energy_ratio_2d(x2d, radius_ratio=0.45):
    x = np.asarray(x2d, dtype=np.float32)
    x = x - x.mean()
    spec = np.fft.fftshift(np.fft.fft2(x))
    power = np.abs(spec) ** 2

    h, w = x.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cy, cx = h // 2, w // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r = r / (r.max() + 1e-8)

    high = r > radius_ratio

    total = power.sum() + 1e-8
    high_power = power[high].sum()
    return float(high_power / total)


def plot_s5(image_3d, base_prob_fg, suff_prob_fg, base_fg, suff_fg, z, case_idx):
    ct = normalize_ct(image_3d[z])

    base_spec = fft_log_spectrum_2d(base_prob_fg[z])
    suff_spec = fft_log_spectrum_2d(suff_prob_fg[z])

    base_hf = high_frequency_energy_ratio_2d(base_prob_fg[z], radius_ratio=D_HIGH)
    suff_hf = high_frequency_energy_ratio_2d(suff_prob_fg[z], radius_ratio=D_HIGH)

    fig, axes = plt.subplots(1, 6, figsize=(16, 3.4))

    axes[0].imshow(ct, cmap="gray")
    axes[0].set_title("Input CT")
    axes[0].axis("off")

    axes[1].imshow(base_prob_fg[z], cmap="magma")
    axes[1].set_title("Final DDIM prob")
    axes[1].axis("off")

    axes[2].imshow(suff_prob_fg[z], cmap="magma")
    axes[2].set_title("SUFF prob")
    axes[2].axis("off")

    axes[3].imshow(base_fg[z], cmap="gray")
    axes[3].set_title("Final DDIM mask")
    axes[3].axis("off")

    axes[4].imshow(base_spec, cmap="magma")
    axes[4].set_title(f"FFT before\nHF={base_hf:.4f}")
    axes[4].axis("off")

    axes[5].imshow(suff_spec, cmap="magma")
    axes[5].set_title(f"FFT after SUFF\nHF={suff_hf:.4f}")
    axes[5].axis("off")

    fig.suptitle(f"Case {case_idx:03d}, z={z}: frequency-domain visualization before/after SUFF", fontsize=11)
    plt.tight_layout()

    out_png = FIG_DIR / f"Fig_S5_SUFF_case{case_idx:03d}_z{z:03d}.png"
    out_pdf = FIG_DIR / f"Fig_S5_SUFF_case{case_idx:03d}_z{z:03d}.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved S5:", out_png)
    print("Saved S5:", out_pdf)

    return base_hf, suff_hf


def main():
    print("=" * 80)
    print("SUFF case-level post-processing")
    print("=" * 80)
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"CKPT_PATH: {CKPT_PATH}")
    print(f"CASE_INDEX: {CASE_INDEX}")
    print(f"DEVICE: {DEVICE}")
    print(f"NOISE_SAMPLES: {NOISE_SAMPLES}")
    print(f"DDIM_STEPS setting expected from sample_diffusion: {DDIM_STEPS}")
    print(f"D_LOW={D_LOW}, D_HIGH={D_HIGH}")
    print("=" * 80)

    seed_everything(BASE_SEED)

    val_ds = get_btcv_val_dataset()
    print("val dataset length:", len(val_ds))

    image, label, item = prepare_case_batch(val_ds, CASE_INDEX)
    print("image shape:", tuple(image.shape))
    print("label shape:", tuple(label.shape))

    target_13 = raw_label_to_13ch(label)

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

    image_device = image.to(DEVICE)

    start_time = time.time()

    with torch.no_grad():
        out_raw = inferer(image_device, predictor)

    elapsed = time.time() - start_time

    print(f"Inference elapsed sec: {elapsed:.2f}")

    out_raw = out_raw.detach().cpu()

    base_prob = torch.sigmoid(out_raw[:, :13])
    suff_prob = torch.sigmoid(out_raw[:, 13:26])

    base_bin = (base_prob > 0.5).float()
    suff_bin = (suff_prob > 0.5).float()

    base_dice, base_hd95, base_dice_cls = compute_metrics(base_bin, target_13)
    suff_dice, suff_hd95, suff_dice_cls = compute_metrics(suff_bin, target_13)

    print("=" * 80)
    print("Metrics")
    print("=" * 80)
    print(f"Final DDIM Dice: {base_dice:.6f}")
    print(f"SUFF Dice:       {suff_dice:.6f}")
    print(f"Dice gain:       {suff_dice - base_dice:.6f}")
    print(f"Final DDIM HD95: {base_hd95}")
    print(f"SUFF HD95:       {suff_hd95}")

    image_3d = image[0, 0].detach().cpu().numpy()
    gt_13 = target_13[0].detach().cpu().numpy().astype(bool)
    gt_fg = gt_13.any(axis=0)

    base_prob_fg, base_fg = foreground_from_prob(base_prob)
    suff_prob_fg, suff_fg = foreground_from_prob(suff_prob)

    z_best, removed_fp, added_fp, diff = select_s3_slice(
        image_3d=image_3d,
        gt_fg=gt_fg,
        base_fg=base_fg,
        suff_fg=suff_fg,
    )

    print("=" * 80)
    print("Difference summary")
    print("=" * 80)
    print("Selected z:", z_best)
    print("GT foreground voxels:", int(gt_fg.sum()))
    print("Final DDIM foreground voxels:", int(base_fg.sum()))
    print("SUFF foreground voxels:", int(suff_fg.sum()))
    print("Changed voxels:", int(diff.sum()))
    print("Removed FP voxels:", int(removed_fp.sum()))
    print("Added FP voxels:", int(added_fp.sum()))

    plot_s3(
        image_3d=image_3d,
        gt_fg=gt_fg,
        base_fg=base_fg,
        suff_fg=suff_fg,
        z=z_best,
        case_idx=CASE_INDEX,
    )

    base_hf, suff_hf = plot_s5(
        image_3d=image_3d,
        base_prob_fg=base_prob_fg,
        suff_prob_fg=suff_prob_fg,
        base_fg=base_fg,
        suff_fg=suff_fg,
        z=z_best,
        case_idx=CASE_INDEX,
    )

    print("=" * 80)
    print("Frequency summary")
    print("=" * 80)
    print(f"FFT before SUFF HF ratio: {base_hf:.6f}")
    print(f"FFT after  SUFF HF ratio: {suff_hf:.6f}")

    out_npz = OUT_DIR / f"suff_case{CASE_INDEX:03d}_final_ddim_vs_suff.npz"
    np.savez_compressed(
        out_npz,
        image_3d=image_3d.astype(np.float32),
        gt_fg=gt_fg.astype(np.uint8),
        base_prob_fg=base_prob_fg.astype(np.float32),
        suff_prob_fg=suff_prob_fg.astype(np.float32),
        base_fg=base_fg.astype(np.uint8),
        suff_fg=suff_fg.astype(np.uint8),
        removed_fp=removed_fp.astype(np.uint8),
        added_fp=added_fp.astype(np.uint8),
        diff=diff.astype(np.uint8),
        z_best=np.array(z_best),
        base_dice=np.array(base_dice),
        suff_dice=np.array(suff_dice),
        base_hd95=np.array(base_hd95),
        suff_hd95=np.array(suff_hd95),
        base_hf=np.array(base_hf),
        suff_hf=np.array(suff_hf),
        base_dice_cls=base_dice_cls,
        suff_dice_cls=suff_dice_cls,
    )

    out_txt = OUT_DIR / f"suff_case{CASE_INDEX:03d}_summary.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("SUFF case-level post-processing summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"case_index: {CASE_INDEX}\n")
        f.write(f"checkpoint: {CKPT_PATH}\n")
        f.write(f"noise_samples: {NOISE_SAMPLES}\n")
        f.write(f"roi_size: {ROI_SIZE}\n")
        f.write(f"overlap: {SW_OVERLAP}\n")
        f.write(f"D_low: {D_LOW}\n")
        f.write(f"D_high: {D_HIGH}\n")
        f.write(f"Butterworth_order: {BUTTERWORTH_ORDER}\n")
        f.write(f"low_freq_weight: {LOW_FREQ_WEIGHT}\n")
        f.write(f"band_freq_weight: {BAND_FREQ_WEIGHT}\n")
        f.write(f"high_freq_weight: {HIGH_FREQ_WEIGHT}\n")
        f.write(f"elapsed_sec: {elapsed:.2f}\n")
        f.write("\nMetrics\n")
        f.write(f"final_ddim_dice: {base_dice:.6f}\n")
        f.write(f"suff_dice: {suff_dice:.6f}\n")
        f.write(f"dice_gain: {suff_dice - base_dice:.6f}\n")
        f.write(f"final_ddim_hd95: {base_hd95}\n")
        f.write(f"suff_hd95: {suff_hd95}\n")
        f.write("\nDifference summary\n")
        f.write(f"selected_z: {z_best}\n")
        f.write(f"gt_foreground_voxels: {int(gt_fg.sum())}\n")
        f.write(f"final_ddim_foreground_voxels: {int(base_fg.sum())}\n")
        f.write(f"suff_foreground_voxels: {int(suff_fg.sum())}\n")
        f.write(f"changed_voxels: {int(diff.sum())}\n")
        f.write(f"removed_fp_voxels: {int(removed_fp.sum())}\n")
        f.write(f"added_fp_voxels: {int(added_fp.sum())}\n")
        f.write("\nFrequency summary\n")
        f.write(f"fft_before_suff_hf_ratio: {base_hf:.6f}\n")
        f.write(f"fft_after_suff_hf_ratio: {suff_hf:.6f}\n")

    print("Saved npz:", out_npz)
    print("Saved summary:", out_txt)
    print("Done.")


if __name__ == "__main__":
    main()