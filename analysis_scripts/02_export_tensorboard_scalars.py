from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import csv


ROOT = Path("/data/oldproject/exp/Diff-UNet-main2")
OUT_DIR = Path("/data/oldproject/analysis_outputs")
SCALAR_DIR = OUT_DIR / "02_exported_scalars"
SUMMARY_CSV = OUT_DIR / "02_scalar_file_summary.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
SCALAR_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    return (
        text.replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
        .replace(" ", "_")
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace(",", "_")
    )


def main():
    event_files = sorted(ROOT.rglob("events.out.tfevents*"))
    summary_rows = []

    for event_file in event_files:
        ea = EventAccumulator(str(event_file))
        ea.Reload()
        scalar_tags = ea.Tags().get("scalars", [])

        if not scalar_tags:
            continue

        rel_path = event_file.relative_to(ROOT)
        rel_str = str(rel_path)

        for tag in scalar_tags:
            events = ea.Scalars(tag)
            if not events:
                continue

            out_name = safe_name(rel_str + "__" + tag) + ".csv"
            out_csv = SCALAR_DIR / out_name

            with out_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["event_file", "relative_event_file", "tag", "wall_time", "step", "value"])
                for e in events:
                    writer.writerow([str(event_file), rel_str, tag, e.wall_time, e.step, e.value])

            values = [float(e.value) for e in events]
            steps = [int(e.step) for e in events]

            summary_rows.append({
                "relative_event_file": rel_str,
                "tag": tag,
                "num_points": len(events),
                "min_step": min(steps),
                "max_step": max(steps),
                "first_value": values[0],
                "last_value": values[-1],
                "min_value": min(values),
                "max_value": max(values),
                "output_csv": str(out_csv),
            })

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "relative_event_file",
            "tag",
            "num_points",
            "min_step",
            "max_step",
            "first_value",
            "last_value",
            "min_value",
            "max_value",
            "output_csv",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Done.")
    print(f"Scalar series exported: {len(summary_rows)}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Scalar CSV folder: {SCALAR_DIR}")


if __name__ == "__main__":
    main()