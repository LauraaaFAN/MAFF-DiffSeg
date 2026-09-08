#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


CASE_INDEX = 9

IN_NPZ = Path(f"/data/oldproject/analysis_outputs/suff_case/suff_case{CASE_INDEX:03d}_final_ddim_vs_suff.npz")
FIG_DIR = Path("/data/oldproject/analysis_outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 如果想手动指定 slice，把 None 改成整数，比如 208
Z_OVERRIDE = None


def normalize_ct(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return np.zeros_like(x)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo)


def overlay(ax, ct, mask, title, alpha=0.55):
    ax.imshow(ct, cmap="gray")
    if mask is not None:
        masked = np.ma.masked_where(mask <= 0, mask)
        purple_cmap = ListedColormap(["#f2c94c"])
        ax.imshow(masked, cmap=purple_cmap, alpha=alpha)       
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    if not IN_NPZ.exists():
        raise FileNotFoundError(IN_NPZ)

    data = np.load(IN_NPZ)

    if Z_OVERRIDE is None:
        z = int(data["z_best"])
    else:
        z = int(Z_OVERRIDE)

    image_3d = data["image_3d"]
    gt_fg = data["gt_fg"].astype(bool)
    base_fg = data["base_fg"].astype(bool)
    suff_fg = data["suff_fg"].astype(bool)

    ct = normalize_ct(image_3d[z])

    fp_before = base_fg & (~gt_fg)
    fp_after = suff_fg & (~gt_fg)
    removed_fp = base_fg & (~suff_fg) & (~gt_fg)
    added_fp = suff_fg & (~base_fg) & (~gt_fg)

    panels = [
        ("Input CT", None),
        ("Ground truth", gt_fg[z]),
        ("Final DDIM pred", base_fg[z]),
        ("SUFF pred", suff_fg[z]),
        ("FP before SUFF", fp_before[z]),
        ("Residual FP after SUFF", fp_after[z]),
        ("Removed FP by SUFF", removed_fp[z]),
        ("Added FP by SUFF", added_fp[z]),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(11.5, 6.0))
    axes = axes.ravel()

    for ax, (title, mask) in zip(axes, panels):
        overlay(ax, ct, mask, title)

    fig.suptitle(
        f"Case {CASE_INDEX:03d}, z={z}: false-positive reduction after SUFF",
        fontsize=11,
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.subplots_adjust(hspace=0.12, wspace=0.02)

    out_png = FIG_DIR / f"Fig_S5_false_positive_2row_case{CASE_INDEX:03d}_z{z:03d}.png"
    out_pdf = FIG_DIR / f"Fig_S5_false_positive_2row_case{CASE_INDEX:03d}_z{z:03d}.pdf"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", out_pdf)
    print("Selected z:", z)
    print("FP before SUFF on selected slice:", int(fp_before[z].sum()))
    print("Residual FP after SUFF on selected slice:", int(fp_after[z].sum()))
    print("Removed FP by SUFF on selected slice:", int(removed_fp[z].sum()))
    print("Added FP by SUFF on selected slice:", int(added_fp[z].sum()))


if __name__ == "__main__":
    main()