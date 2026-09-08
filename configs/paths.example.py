from pathlib import Path

# Modify these paths according to your local environment.
PROJECT_ROOT = Path("/path/to/MAFF-DiffSeg")
BTCV_ROOT = PROJECT_ROOT / "BTCV"

DATA_DIR = BTCV_ROOT / "RawData" / "Training"
IMAGE_DIR = DATA_DIR / "imagesTr"
LABEL_DIR = DATA_DIR / "labelsTr"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
