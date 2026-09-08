import csv
import time
from pathlib import Path
from importlib.machinery import SourceFileLoader

import numpy as np
import torch
from monai.inferers import SlidingWindowInferer


SMOKE_SCRIPT = "/data/oldproject/analysis_scripts/05_inference_smoke_test.py"
smoke = SourceFileLoader("smoke", SMOKE_SCRIPT).load_module()


BTCV_ROOT = Path("/data/oldproject/exp/Diff-UNet-main2/BTCV")
DATA_DIR = str(BTCV_ROOT / "RawData/Training") + "/"
CHECKPOINT = BTCV_ROOT / "logs_btcv/best_model_0.7680.pt"

OUT_DIR = Path("/data/oldproject/analysis_outputs/noise_sample_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CASE_CSV = OUT_DIR / "noise_sample_ablation_case_metrics.csv"
OUT_SUMMARY_CSV = OUT_DIR / "noise_sample_ablation_summary.csv"
OUT_SUMMARY_TXT = OUT_DIR / "noise_sample_ablation_summary.txt"

DEVICE = "cuda:0"
ROI_SIZE = [96, 96, 96]
DDIM_STEPS = 10

NOISE_SAMPLE_LIST = [1, 3, 5, 7, 10]
BASE_SEED = 20260818


def load_checkpoint_into_model():
    model = smoke.DiffUNet(ddim_steps=DDIM_STEPS).to(DEVICE)

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

    print("Loaded checkpoint.")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")

    if len(missing) > 0 or len(unexpected) > 0:
        print("WARNING: checkpoint keys are not fully matched.")
        print("First missing:", missing[:10])
        print("First unexpected:", unexpected[:10])

    model.eval()
    return model


def append_case_rows(rows):
    file_exists = OUT_CASE_CSV.exists()

    fieldnames = [
        "noise_samples",
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
        "noise_samples",
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
        f.write("Noise sample ablation summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"DDIM steps fixed at {DDIM_STEPS}\n")
        f.write("Aggregation: average raw model outputs across n independent DDIM samplings, then sigmoid + threshold.\n")
        f.write("=" * 80 + "\n")
        for row in summary_rows:
            f.write(
                f"n = {row['noise_samples']}: "
                f"Dice = {row['overall_mean_dice']:.6f} ± {row['overall_std_dice']:.6f}, "
                f"HD95 = {row['overall_mean_hd95']:.6f} ± {row['overall_std_hd95']:.6f}, "
                f"elapsed_sec = {row['elapsed_sec']:.2f}, "
                f"peak_mem_gb = {row['peak_cuda_memory_gb']:.4f}\n"
            )


def run_one_n_setting(noise_samples, val_ds, model):
    print()
    print("#" * 100)
    print(f"Running noise samples n = {noise_samples}")
    print("#" * 100)

    inferer = SlidingWindowInferer(
        roi_size=ROI_SIZE,
        sw_batch_size=1,
        overlap=0.6,
    )

    case_mean_dices = []
    case_mean_hd95s = []

    start_n = time.time()
    torch.cuda.reset_peak_memory_stats()

    for case_idx in range(len(val_ds)):
        print()
        print("=" * 80)
        print(f"n = {noise_samples} | case {case_idx + 1}/{len(val_ds)}")

        start_case = time.time()

        batch = val_ds[case_idx]

        image = smoke.ensure_batch_dim(batch["image"]).float().to(DEVICE)

        # Match original train.py validation_step:
        # use batch["label"], not batch["raw_label"].
        if "label" not in batch:
            raise RuntimeError("No label found in batch.")
        label = smoke.ensure_batch_dim(batch["label"]).to(DEVICE)
        target = smoke.convert_labels(label)

        output_accum = None

        with torch.no_grad():
            for sample_idx in range(noise_samples):
                seed = BASE_SEED + case_idx * 1000 + sample_idx

                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                output = inferer(image, model, pred_type="ddim_sample")

                if output_accum is None:
                    output_accum = output.detach()
                else:
                    output_accum = output_accum + output.detach()

                del output
                torch.cuda.empty_cache()

            output_mean = output_accum / float(noise_samples)
            prob = torch.sigmoid(output_mean)
            pred_binary = (prob > 0.5).float().cpu()

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
            "noise_samples": noise_samples,
            "ddim_steps": DDIM_STEPS,
            "case_idx": case_idx,
            "case_mean_dice": case_mean_dice,
            "case_mean_hd95": case_mean_hd95,
            "elapsed_sec": elapsed_case,
        }

        for i, x in enumerate(dices, start=1):
            row[f"dice_class_{i}"] = x

        for i, x in enumerate(hd95s, start=1):
            row[f"hd95_class_{i}"] = x

        append_case_rows([row])

        del image, label, target, output_accum, output_mean, prob, pred_binary, pred_np, target_np
        torch.cuda.empty_cache()

    elapsed_n = time.time() - start_n
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3

    summary = {
        "noise_samples": noise_samples,
        "ddim_steps": DDIM_STEPS,
        "num_cases": len(val_ds),
        "overall_mean_dice": float(np.mean(case_mean_dices)),
        "overall_std_dice": float(np.std(case_mean_dices)),
        "overall_mean_hd95": float(np.mean(case_mean_hd95s)),
        "overall_std_hd95": float(np.std(case_mean_hd95s)),
        "elapsed_sec": elapsed_n,
        "peak_cuda_memory_gb": peak_mem_gb,
    }

    print()
    print(f"Summary for n = {noise_samples}")
    print(f"  overall_mean_dice: {summary['overall_mean_dice']:.4f}")
    print(f"  overall_std_dice:  {summary['overall_std_dice']:.4f}")
    print(f"  overall_mean_hd95: {summary['overall_mean_hd95']:.4f}")
    print(f"  elapsed_sec:       {elapsed_n:.2f}")
    print(f"  peak_mem_gb:       {peak_mem_gb:.2f}")

    return summary


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("===== Noise sample ablation started =====")
    print(f"Data dir: {DATA_DIR}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Output dir: {OUT_DIR}")
    print(f"DDIM steps fixed at: {DDIM_STEPS}")
    print(f"Noise sample list: {NOISE_SAMPLE_LIST}")
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

    model = load_checkpoint_into_model()

    all_summaries = []
    start_all = time.time()

    for noise_samples in NOISE_SAMPLE_LIST:
        summary = run_one_n_setting(noise_samples, val_ds, model)
        all_summaries.append(summary)
        save_summary(all_summaries)

    elapsed_all = time.time() - start_all

    print()
    print("===== All noise sample settings finished =====")
    print(f"Total elapsed sec: {elapsed_all:.2f}")
    print(f"Case metrics CSV: {OUT_CASE_CSV}")
    print(f"Summary CSV: {OUT_SUMMARY_CSV}")
    print(f"Summary TXT: {OUT_SUMMARY_TXT}")

    print()
    print("Final summary:")
    for row in all_summaries:
        print(
            f"  n={row['noise_samples']:>2} | "
            f"Dice={row['overall_mean_dice']:.4f} ± {row['overall_std_dice']:.4f} | "
            f"HD95={row['overall_mean_hd95']:.4f} ± {row['overall_std_hd95']:.4f} | "
            f"time={row['elapsed_sec']:.1f}s"
        )


if __name__ == "__main__":
    main()