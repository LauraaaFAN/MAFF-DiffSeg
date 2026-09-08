import csv
import time
from pathlib import Path
from importlib.machinery import SourceFileLoader

import numpy as np
import torch
from monai.inferers import SlidingWindowInferer


# Reuse definitions from smoke test script.
SMOKE_SCRIPT = "/data/oldproject/analysis_scripts/05_inference_smoke_test.py"
smoke = SourceFileLoader("smoke", SMOKE_SCRIPT).load_module()


BTCV_ROOT = Path("/data/oldproject/exp/Diff-UNet-main2/BTCV")
DATA_DIR = str(BTCV_ROOT / "RawData/Training") + "/"
CHECKPOINT = BTCV_ROOT / "logs_btcv/best_model_0.7680.pt"

OUT_DIR = Path("/data/oldproject/analysis_outputs/ddim_step_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CASE_CSV = OUT_DIR / "ddim_step_ablation_case_metrics.csv"
OUT_SUMMARY_CSV = OUT_DIR / "ddim_step_ablation_summary.csv"
OUT_SUMMARY_TXT = OUT_DIR / "ddim_step_ablation_summary.txt"

DEVICE = "cuda:0"
ROI_SIZE = [96, 96, 96]

DDIM_STEPS_LIST = [5, 10, 15, 20, 25]


def load_checkpoint_into_model(ddim_steps):
    model = smoke.DiffUNet(ddim_steps=ddim_steps).to(DEVICE)

    sd = torch.load(CHECKPOINT, map_location="cpu")
    if isinstance(sd, dict):
        if "module" in sd:
            sd = sd["module"]
        elif "state_dict" in sd:
            sd = sd["state_dict"]
        elif "model" in sd:
            sd = sd["model"]

    new_sd = {}
    for k, v in sd.items():
        k = str(k)
        if k.startswith("module."):
            k = k[7:]
        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    print(f"Loaded checkpoint for DDIM steps = {ddim_steps}")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")

    if len(missing) > 0 or len(unexpected) > 0:
        print("WARNING: checkpoint keys are not fully matched.")
        print("First missing:", missing[:10])
        print("First unexpected:", unexpected[:10])

    model.eval()
    return model


def run_one_step_setting(ddim_steps, val_ds):
    print()
    print("#" * 100)
    print(f"Running DDIM steps = {ddim_steps}")
    print("#" * 100)

    model = load_checkpoint_into_model(ddim_steps)

    inferer = SlidingWindowInferer(
        roi_size=ROI_SIZE,
        sw_batch_size=1,
        overlap=0.6,
    )

    case_rows = []
    case_mean_dices = []
    case_mean_hd95s = []

    start_step = time.time()
    torch.cuda.reset_peak_memory_stats()

    for case_idx in range(len(val_ds)):
        print()
        print("=" * 80)
        print(f"DDIM steps {ddim_steps} | case {case_idx + 1}/{len(val_ds)}")

        start_case = time.time()

        batch = val_ds[case_idx]

        image = smoke.ensure_batch_dim(batch["image"]).float().to(DEVICE)

        # Match original train.py validation_step:
        # use batch["label"], not batch["raw_label"].
        if "label" not in batch:
            raise RuntimeError("No label found in batch.")
        label = smoke.ensure_batch_dim(batch["label"]).to(DEVICE)
        target = smoke.convert_labels(label)

        with torch.no_grad():
            output = inferer(image, model, pred_type="ddim_sample")
            output = torch.sigmoid(output)
            pred_binary = (output > 0.5).float().cpu()

        pred_np = pred_binary.numpy()
        target_np = target.cpu().numpy()

        dices, hd95s = smoke.compute_metrics(pred_np, target_np)

        case_mean_dice = float(np.mean(dices))
        case_mean_hd95 = float(np.mean(hd95s))
        elapsed_case = time.time() - start_case

        case_mean_dices.append(case_mean_dice)
        case_mean_hd95s.append(case_mean_hd95)

        print(f"case_mean_dice: {case_mean_dice:.4f}")
        print(f"case_mean_hd95: {case_mean_hd95:.4f}")
        print(f"elapsed_case_sec: {elapsed_case:.2f}")

        row = {
            "ddim_steps": ddim_steps,
            "case_idx": case_idx,
            "case_mean_dice": case_mean_dice,
            "case_mean_hd95": case_mean_hd95,
            "elapsed_sec": elapsed_case,
        }

        for i, x in enumerate(dices, start=1):
            row[f"dice_class_{i}"] = x

        for i, x in enumerate(hd95s, start=1):
            row[f"hd95_class_{i}"] = x

        case_rows.append(row)

        # Save partial results after every case, useful for overnight runs.
        append_case_rows([row])

        del image, label, target, output, pred_binary, pred_np, target_np
        torch.cuda.empty_cache()

    elapsed_step = time.time() - start_step
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3

    step_summary = {
        "ddim_steps": ddim_steps,
        "num_cases": len(val_ds),
        "overall_mean_dice": float(np.mean(case_mean_dices)),
        "overall_std_dice": float(np.std(case_mean_dices)),
        "overall_mean_hd95": float(np.mean(case_mean_hd95s)),
        "overall_std_hd95": float(np.std(case_mean_hd95s)),
        "elapsed_sec": elapsed_step,
        "peak_cuda_memory_gb": peak_mem_gb,
    }

    print()
    print(f"Summary for DDIM steps = {ddim_steps}")
    print(f"  overall_mean_dice: {step_summary['overall_mean_dice']:.4f}")
    print(f"  overall_std_dice:  {step_summary['overall_std_dice']:.4f}")
    print(f"  overall_mean_hd95: {step_summary['overall_mean_hd95']:.4f}")
    print(f"  elapsed_sec:       {elapsed_step:.2f}")
    print(f"  peak_mem_gb:       {peak_mem_gb:.2f}")

    del model
    torch.cuda.empty_cache()

    return step_summary


def append_case_rows(rows):
    file_exists = OUT_CASE_CSV.exists()

    fieldnames = [
        "ddim_steps",
        "case_idx",
        "case_mean_dice",
        "case_mean_hd95",
        "elapsed_sec",
    ]

    for i in range(1, 14):
        fieldnames.append(f"dice_class_{i}")
    for i in range(1, 14):
        fieldnames.append(f"hd95_class_{i}")

    with OUT_CASE_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def save_summary(summary_rows):
    fieldnames = [
        "ddim_steps",
        "num_cases",
        "overall_mean_dice",
        "overall_std_dice",
        "overall_mean_hd95",
        "overall_std_hd95",
        "elapsed_sec",
        "peak_cuda_memory_gb",
    ]

    with OUT_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with OUT_SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write("DDIM step ablation summary\n")
        f.write("=" * 80 + "\n")
        for row in summary_rows:
            f.write(
                f"DDIM steps = {row['ddim_steps']}: "
                f"Dice = {row['overall_mean_dice']:.6f} ± {row['overall_std_dice']:.6f}, "
                f"HD95 = {row['overall_mean_hd95']:.6f} ± {row['overall_std_hd95']:.6f}, "
                f"elapsed_sec = {row['elapsed_sec']:.2f}, "
                f"peak_mem_gb = {row['peak_cuda_memory_gb']:.4f}\n"
            )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("===== DDIM step ablation started =====")
    print(f"Data dir: {DATA_DIR}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Output dir: {OUT_DIR}")
    print(f"DDIM steps list: {DDIM_STEPS_LIST}")
    print()

    # Remove old outputs to avoid mixing runs.
    if OUT_CASE_CSV.exists():
        OUT_CASE_CSV.unlink()
    if OUT_SUMMARY_CSV.exists():
        OUT_SUMMARY_CSV.unlink()
    if OUT_SUMMARY_TXT.exists():
        OUT_SUMMARY_TXT.unlink()

    train_ds, val_ds, test_ds = smoke.get_loader_btcv(data_dir=DATA_DIR)
    print(f"val_ds length: {len(val_ds)}")

    all_summaries = []
    start_all = time.time()

    for ddim_steps in DDIM_STEPS_LIST:
        summary = run_one_step_setting(ddim_steps, val_ds)
        all_summaries.append(summary)
        save_summary(all_summaries)

    elapsed_all = time.time() - start_all

    print()
    print("===== All DDIM step settings finished =====")
    print(f"Total elapsed sec: {elapsed_all:.2f}")
    print(f"Case metrics CSV: {OUT_CASE_CSV}")
    print(f"Summary CSV: {OUT_SUMMARY_CSV}")
    print(f"Summary TXT: {OUT_SUMMARY_TXT}")

    print()
    print("Final summary:")
    for row in all_summaries:
        print(
            f"  steps={row['ddim_steps']:>2} | "
            f"Dice={row['overall_mean_dice']:.4f} ± {row['overall_std_dice']:.4f} | "
            f"HD95={row['overall_mean_hd95']:.4f} ± {row['overall_std_hd95']:.4f} | "
            f"time={row['elapsed_sec']:.1f}s"
        )


if __name__ == "__main__":
    main()