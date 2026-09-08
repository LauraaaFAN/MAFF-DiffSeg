from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


IN_CSV = Path("/data/oldproject/analysis_outputs/noise_sample_ablation/noise_sample_ablation_summary.csv")

OUT_DIR = Path("/data/oldproject/analysis_outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Fig_S2d_Noise_samples_Dice.png"
OUT_PDF = OUT_DIR / "Fig_S2d_Noise_samples_Dice.pdf"


def main():
    df = pd.read_csv(IN_CSV)
    df = df.sort_values("noise_samples")

    x = df["noise_samples"]
    y = df["overall_mean_dice"]
    yerr = df["overall_std_dice"]

    plt.figure(figsize=(4.2, 3.4))

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        linewidth=1.8,
        markersize=5,
        capsize=3,
    )

    plt.xlabel("Number of noise samples")
    plt.ylabel("Mean Dice")
    plt.xticks([1, 3, 5, 7, 10])
    plt.ylim(0.5, 0.9)

    # Mark the selected default setting.
    plt.axvline(x=5, linestyle="--", linewidth=1)
    plt.text(5.25, ymax + 0.002, "Selected n=5", fontsize=9, va="top")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")

    print("Saved:")
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()