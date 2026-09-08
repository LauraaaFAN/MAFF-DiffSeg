from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


IN_CSV = Path("/data/oldproject/analysis_outputs/ddim_step_ablation/ddim_step_ablation_summary.csv")

OUT_DIR = Path("/data/oldproject/analysis_outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Fig_S2c_DDIM_steps_Dice.png"
OUT_PDF = OUT_DIR / "Fig_S2c_DDIM_steps_Dice.pdf"


def main():
    df = pd.read_csv(IN_CSV)

    # Sort by DDIM steps to ensure correct x-axis order.
    df = df.sort_values("ddim_steps")

    x = df["ddim_steps"]
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

    plt.xlabel("DDIM sampling steps")
    plt.ylabel("Mean Dice")
    plt.title("Effect of DDIM sampling steps")
    plt.ylim(0.5, 0.9)
    plt.xticks([5, 10, 15, 20, 25])

    # Mark the selected default setting.
    plt.axvline(x=10, linestyle="--", linewidth=1)
    plt.text(10.5, 0.805, "Selected T=10", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")

    print("Saved:")
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()