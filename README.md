# MAFF-DiffSeg

This repository contains the implementation and supplementary analysis scripts for a diffusion-based 3D medical image segmentation framework evaluated on the BTCV abdominal CT dataset.

## Overview

The implemented model consists of:

- a `BasicUNetEncoder` for extracting multi-scale image-conditioned CT features;
- a denoising `BasicUNetDe` network for predicting 13-channel segmentation logits during diffusion sampling;
- supplementary scripts for training-curve visualization, DDIM sampling-step analysis, noise-sample analysis, SUFF post-processing, false-positive visualization, frequency-domain visualization, and per-organ metric calculation.

## Repository structure

```text
MAFF-DiffSeg/
├── BTCV/
│   ├── train.py
│   ├── test.py
│   ├── unet/
│   ├── guided_diffusion/
│   ├── dataset/
│   └── light_training/
├── analysis_scripts/
├── configs/
├── docs/
├── requirements.txt
└── README.md
```

## Dataset

The BTCV dataset is not included in this repository. Please download the dataset from the official source and organize it as:

```
BTCV/RawData/Training/
├── imagesTr/
└── labelsTr/
```

## Environment

Install the required Python packages:

```Bash
pip install -r requirements.txt
```

## Training

Example:

```Bash
cd BTCV
python train.py
```

Please modify the dataset path and output path according to your local environment.

## Supplementary analysis

Example scripts are provided in `analysis_scripts/`.

DDIM sampling-step analysis:

```Bash
python analysis_scripts/07_run_ddim_steps_ablation.py
```

Noise-sample analysis:

```Bash
python analysis_scripts/09_run_noise_sample_ablation.py
```

SUFF validation analysis for Table S2:

```Bash
python analysis_scripts/26_run_suff_validation_table_s2.py
```

Per-organ Dice and HD95 analysis for Table S3:

```Bash
python analysis_scripts/28_run_table_s3_per_organ_metrics.py
```

Model parameter-count analysis:

```Bash
python analysis_scripts/29_count_model_params_flops.py
```

## Notes

This repository does not include raw medical images, private clinical data, trained checkpoints, TensorBoard logs, or intermediate prediction volumes.