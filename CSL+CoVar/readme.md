# CSL

## Getting Started

### Installation

```bash
cd CSL
conda create -n csl python=3.11.9
conda activate csl
pip install -r requirements.txt
pip install torch==2.4.0 torchvision==0.19.0 -f https://download.pytorch.org/whl/torch_stable.html

```

### Dataset

- Pascal: (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)
- Cityscapes: (https://www.cityscapes-dataset.com/)

Please modify your dataset path in configuration files.

```
├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass
    
├── [Your Cityscapes Path]
    ├── leftImg8bit
    └── gtFine
```

## Usage

### csl

```bash
cd CSL
scripts/csl.sh
```

Thresholding strategy options for pseudo-label selection:

- Dynamic thresholding (default): uses max confidence + residual variance with class-wise statistics.
- Fixed thresholding: uses only max confidence with a hard threshold (default 0.95).

Example for fixed thresholding:

```bash
python CSL.py ... --threshold_strategy fixed --fixed_threshold 0.95
```

If you want to export pseudo-label diagnostic samples during training, add:

```bash
python CSL.py ... --enable_visual_artifacts
python scripts/pseudo_label_artifacts.py --input_dir /path/to/save_path/pseudo_diagnostics
```

The training flag is disabled by default. When enabled, training saves sampled pseudo-label diagnostic data under `save_path/pseudo_diagnostics`; the standalone script renders plots from those saved artifacts after training.

After training completes, CSL now also writes pseudo-label summary metrics to:

```bash
save_path/pseudo_label_metrics_summary.json
```

This summary includes:

- total generated pseudo-label pixels
- total selected pseudo-label pixels
- pseudo-label selection rate (selected / generated)
- pseudo-label accuracy on validation set (pixel accuracy)

The same values are appended to `output.log` at the end of training.

During training, each epoch now also writes an intermediate cumulative summary to `output.log` for trend tracking:

- generated pseudo-label pixels in this epoch
- selected pseudo-label pixels in this epoch
- epoch selection rate
- cumulative generated pseudo-label pixels
- cumulative selected pseudo-label pixels
- cumulative selection rate

These per-epoch trend records are also saved under `per_epoch_cumulative` in `pseudo_label_metrics_summary.json`.

### Supervised Baseline

```bash
cd CSL
scripts/supervised.sh
```