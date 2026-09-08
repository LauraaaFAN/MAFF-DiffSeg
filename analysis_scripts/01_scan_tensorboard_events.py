from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path


ROOT = Path("/data/oldproject/exp/Diff-UNet-main2")
OUT_DIR = Path("/data/oldproject/analysis_outputs")
OUT_TXT = OUT_DIR / "01_tensorboard_event_scan_summary.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def scan_event_file(event_file: Path):
    """
    Return scalar tags for one TensorBoard event file.
    If the file cannot be read, return an error message.
    """
    try:
        ea = EventAccumulator(str(event_file))
        ea.Reload()
        tags = ea.Tags()
        scalars = tags.get("scalars", [])
        images = tags.get("images", [])
        tensors = tags.get("tensors", [])
        return {
            "ok": True,
            "scalars": scalars,
            "images": images,
            "tensors": tensors,
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "scalars": [],
            "images": [],
            "tensors": [],
            "error": str(e),
        }


def main():
    event_files = sorted(ROOT.rglob("events.out.tfevents*"))

    lines = []
    lines.append(f"ROOT: {ROOT}")
    lines.append(f"Total event files: {len(event_files)}")
    lines.append("")

    non_empty_count = 0
    error_count = 0

    for event_file in event_files:
        result = scan_event_file(event_file)
        size_bytes = event_file.stat().st_size
        rel_path = event_file.relative_to(ROOT)

        has_content = bool(result["scalars"] or result["images"] or result["tensors"])

        if not result["ok"]:
            error_count += 1
            lines.append("=" * 100)
            lines.append(f"ERROR FILE: {rel_path}")
            lines.append(f"SIZE: {size_bytes} bytes")
            lines.append(f"ERROR: {result['error']}")
            continue

        if has_content:
            non_empty_count += 1
            lines.append("=" * 100)
            lines.append(f"FILE: {rel_path}")
            lines.append(f"FULL_PATH: {event_file}")
            lines.append(f"SIZE: {size_bytes} bytes")
            lines.append(f"SCALARS: {result['scalars']}")
            lines.append(f"IMAGES: {result['images']}")
            lines.append(f"TENSORS: {result['tensors']}")

    lines.append("")
    lines.append("=" * 100)
    lines.append(f"Non-empty event files: {non_empty_count}")
    lines.append(f"Error event files: {error_count}")
    lines.append(f"Output saved to: {OUT_TXT}")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print(f"Done.")
    print(f"Total event files: {len(event_files)}")
    print(f"Non-empty event files: {non_empty_count}")
    print(f"Error event files: {error_count}")
    print(f"Output saved to: {OUT_TXT}")


if __name__ == "__main__":
    main()