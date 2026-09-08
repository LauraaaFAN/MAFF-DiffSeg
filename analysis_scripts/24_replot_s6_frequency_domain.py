#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASE_INDEX = 9
D_HIGH = 0.45

IN_NPZ = Path(f"/data/oldproject/analysis_outputs/suff_case/suff_case{CASE_INDEX:03d}_final_ddim_vs_suff.npz")
FIG_DIR = Path("/data/oldproject/analysis_outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 如果想手动改层面，就把 None 改成整数，比如 120
Z_OVERRIDE = None


def normalize_ct(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return np.zeros_like(x)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo)


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
    return float(power[high].sum() / (power.sum() + 1e-8))


def main():
    if not IN_NPZ.exists():
        raise FileNotFoundError(IN_NPZ)

    data = np.load(IN_NPZ)

    if Z_OVERRIDE is None:
        z = int(data["z_best"])
    else:
        z = int(Z_OVERRIDE)

    image_3d = data["image_3d"]
    base_prob = data["base_prob_fg"]
    suff_prob = data["suff_prob_fg"]

    ct = normalize_ct(image_3d[z])

    base_prob_2d = base_prob[z]
    suff_prob_2d = suff_prob[z]

    base_spec = fft_log_spectrum_2d(base_prob_2d)
    suff_spec = fft_log_spectrum_2d(suff_prob_2d)

    base_hf = high_frequency_energy_ratio_2d(base_prob_2d, radius_ratio=D_HIGH)
    suff_hf = high_frequency_energy_ratio_2d(suff_prob_2d, radius_ratio=D_HIGH)

    fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.4))

    axes[0].imshow(ct, cmap="gray")
    axes[0].set_title("Input CT", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(base_prob_2d, cmap="magma")
    axes[1].set_title("Final DDIM prob", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(suff_prob_2d, cmap="magma")
    axes[2].set_title("SUFF prob", fontsize=9)
    axes[2].axis("off")

    axes[3].imshow(base_spec, cmap="magma")
    axes[3].set_title(f"FFT before SUFF\nHF={base_hf:.4f}", fontsize=9)
    axes[3].axis("off")

    axes[4].imshow(suff_spec, cmap="magma")
    axes[4].set_title(f"FFT after SUFF\nHF={suff_hf:.4f}", fontsize=9)
    axes[4].axis("off")

    fig.suptitle(
        f"Case {CASE_INDEX:03d}, z={z}: probability and frequency-domain visualization before/after SUFF",
        fontsize=11,
        y=1.03
    )
    plt.tight_layout()

    out_png = FIG_DIR / f"Fig_S5_SUFF_simple_case{CASE_INDEX:03d}_z{z:03d}.png"
    out_pdf = FIG_DIR / f"Fig_S5_SUFF_simple_case{CASE_INDEX:03d}_z{z:03d}.pdf"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("=" * 80)
    print("S5 simple replot done")
    print("=" * 80)
    print("Selected z:", z)
    print(f"Final DDIM Dice: {float(data['base_dice']):.6f}")
    print(f"SUFF Dice:       {float(data['suff_dice']):.6f}")
    print(f"FFT before SUFF HF ratio: {base_hf:.6f}")
    print(f"FFT after  SUFF HF ratio: {suff_hf:.6f}")
    print("Saved:", out_png)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    main()