import csv
from pathlib import Path
import matplotlib.pyplot as plt


# ===== Input files =====
TRAIN_LOSS_CSV = Path(
    "/data/oldproject/analysis_outputs/02_exported_scalars/"
    "BTCV__logs_btcv__diffunet_transunet_datasettings__"
    "events.out.tfevents.1741166515.QinLab3090.1001588.0__train_loss.csv"
)

MEAN_DICE_CSV = Path(
    "/data/oldproject/analysis_outputs/02_exported_scalars/"
    "BTCV__logs_btcv__diffunet_transunet_datasettings__"
    "events.out.tfevents.1741166515.QinLab3090.1001588.0__mean_dice.csv"
)

# ===== Output files =====
OUT_DIR = Path("/data/oldproject/analysis_outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Fig_S2_ab_training_convergence.png"
OUT_PDF = OUT_DIR / "Fig_S2_ab_training_convergence.pdf"


MAX_EPOCH = 3000


def read_scalar_csv(path):
    steps = []
    values = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(float(row["step"]))
            values.append(float(row["value"]))

    return steps, values


def moving_average(values, window=100):
    """Simple moving average for smoothing noisy training loss."""
    if window <= 1 or len(values) < window:
        return values

    smoothed = []
    half = window // 2

    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        smoothed.append(sum(values[start:end]) / (end - start))

    return smoothed


def main():
    train_steps, train_loss = read_scalar_csv(TRAIN_LOSS_CSV)
    dice_epochs, mean_dice = read_scalar_csv(MEAN_DICE_CSV)

    # train_loss was recorded by global_step.
    # Convert global_step to approximate epoch using max global step = 54000 and max epoch = 3000.
    max_train_step = max(train_steps)
    train_epochs = [s / max_train_step * MAX_EPOCH for s in train_steps]

    train_loss_smooth = moving_average(train_loss, window=100)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ---- Fig. S2(a): Training loss ----
    axes[0].plot(train_epochs, train_loss_smooth, linewidth=1.8)
    axes[0].set_title("(a) Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[0].set_xlim(0, MAX_EPOCH)

    # ---- Fig. S2(b): Validation Dice ----
    axes[1].plot(dice_epochs, mean_dice, marker="o", linewidth=1.8, markersize=3.5)
    axes[1].set_title("(b) Validation Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Dice")
    axes[1].set_xlim(0, MAX_EPOCH)
    axes[1].set_ylim(0, 0.85)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    print("Saved:")
    print(OUT_PNG)
    print(OUT_PDF)
    print()
    print("Training loss:")
    print(f"  points: {len(train_loss)}")
    print(f"  first: {train_loss[0]:.4f}")
    print(f"  last:  {train_loss[-1]:.4f}")
    print(f"  min:   {min(train_loss):.4f}")
    print()
    print("Validation mean Dice:")
    print(f"  points: {len(mean_dice)}")
    print(f"  first: {mean_dice[0]:.4f}")
    print(f"  last:  {mean_dice[-1]:.4f}")
    print(f"  max:   {max(mean_dice):.4f}")


if __name__ == "__main__":
    main()