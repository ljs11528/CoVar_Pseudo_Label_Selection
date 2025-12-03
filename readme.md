# CoVar: Semi-Supervised Semantic Segmentation with Reliable Pseudo-Label Selection

This repository contains the implementation of **CoVar**, a method for semi-supervised semantic segmentation with reliable pseudo-label selection.  
CoVar models covariate consistency and prediction uncertainty to assess the reliability of pseudo-labels and selectively use them during training, improving robustness and accuracy under limited annotations.

## Repository Structure

The repository is organized into three main subdirectories. Each subdirectory integrates CoVar into a different existing semi-supervised segmentation method:

- `CSL+CoVar/`: CoVar integrated into the **CSL** framework  
- `SimPLE+CoVar/`: CoVar integrated into **SimPLE**  
- `UniMatch-V2+CoVar/`: CoVar integrated into **UniMatch-V2**  

Each subdirectory typically contains:

- `dataset/`: dataset loading and preprocessing
- `model/`: network architectures and model builders
- `train/`: training logic (supervised / semi-supervised training, pseudo-label generation and selection)
- `configs/`: experiment configuration files (YAML)
- `*.py`: entry scripts for training (e.g., `CSL.py`)

Below we use `CSL+CoVar` as an example. `SimPLE+CoVar` and `UniMatch-V2+CoVar` follow a similar workflow with method-specific configs and scripts.

---

## Environment

We recommend Python 3.8+.

Example using Conda:

```bash
conda create -n covar python=3.8 -y
conda activate covar

pip install -r requirements.txt
# If there is no requirements.txt, install at least:
# pip install torch torchvision
# pip install "pytorch-lightning==2.*"
# pip install pyyaml
# plus your dataset- / visualization-related dependencies
```

Please install a PyTorch + CUDA combination that matches your hardware and driver.

---

## Data Preparation

Dataset roots and splits are specified in the YAML config files under `configs/`.  
Using `CSL+CoVar` as an example:

1. Download the original dataset(s), such as PASCAL VOC or Cityscapes, and organize them like:

```text
CSL+CoVar/
  CSL_new/
    data/
      VOCdevkit/
      ...
```

2. Prepare image ID list files for labeled / unlabeled / validation samples:

- `labeled_id.txt`: IDs of labeled samples
- `unlabeled_id.txt`: IDs of unlabeled samples
- `val_id.txt`: IDs of validation samples

You pass these file paths via command-line arguments when launching training (see below).

---

## Training (Example: CSL+CoVar)

`CSL+CoVar/CSL.py` is the main training entry point for CSL+CoVar.  
Its core responsibilities include:

- Reading model and training hyperparameters from a YAML config
- Building the segmentation model with the CoVar pseudo-label selection mechanism
- Constructing labeled / unlabeled / validation datasets via `SemiDataset`
- Creating PyTorch `DataLoader`s
- Running training with `pytorch_lightning.Trainer`, including checkpointing and logging

### Single / Multi-GPU Training

From the project root:

```bash
cd CSL+CoVar

python CSL.py \
  --config configs/csl_covar_voc.yml \
  --labeled_id_path path/to/labeled_id.txt \
  --unlabeled_id_path path/to/unlabeled_id.txt \
  --val_id_path path/to/val_id.txt \
  --save_path path/to/save_dir
```

Key arguments:

- `--config`: path to the YAML config file  
- `--labeled_id_path`: path to the labeled IDs file  
- `--unlabeled_id_path`: path to the unlabeled IDs file  
- `--val_id_path`: path to the validation IDs file  
- `--save_path`: directory for logs and checkpoints  

Inside `CSL.py`, training is managed by a `pytorch_lightning.Trainer`, typically configured to support:

- Multi-GPU training (e.g., `accelerator="gpu"`, `strategy="ddp_find_unused_parameters_false"`)
- Mixed precision (e.g., `precision="bf16-mixed"`)
- Automatic checkpointing and resume (from `--save_path/checkpoints`)
- TensorBoard logging (under `--save_path/logs`)

In containerized or memory-constrained environments, you can tune the data loader behavior via environment variables:

```bash
export NUM_WORKERS=4
export PIN_MEMORY=1
```

---

## CoVar: Pseudo-Label Selection (Concept)

CoVar is designed to improve the reliability of pseudo-labels in semi-supervised semantic segmentation:

1. Generate pseudo-labels on unlabeled data using a teacher / EMA / current model.
2. For each pixel (or region / sample), compute covariate features such as:
   - prediction confidence
   - class distribution statistics
   - consistency across augmentations or model snapshots
   - uncertainty estimates
3. Use a covariate-driven strategy (CoVar) to:
   - filter out unreliable pseudo-labels, or
   - down-weight them during training
4. Integrate this selection mechanism into existing frameworks (CSL, SimPLE, UniMatch-V2), making them more robust across datasets and label ratios.

For detailed derivations and experimental results, please refer to our paper (add link / BibTeX here when available).

---

## Integration into Other Methods

### SimPLE+CoVar

`SimPLE+CoVar/` integrates the CoVar pseudo-label selection into the **SimPLE** semi-supervised segmentation framework.

Typical usage pattern:

```bash
cd SimPLE+CoVar

python main_simple_covar.py \
  --config configs/simple_covar_voc.yml \
  --labeled_id_path path/to/labeled_id.txt \
  --unlabeled_id_path path/to/unlabeled_id.txt \
  --val_id_path path/to/val_id.txt \
  --save_path path/to/save_dir
```

(Replace the script and config names above with the actual ones used in your implementation.)

The directory structure and configuration style are similar to `CSL+CoVar`, but adapted to the SimPLE training pipeline.

### UniMatch-V2+CoVar

`UniMatch-V2+CoVar/` integrates CoVar into the **UniMatch-V2** framework.

Example (adapt to your actual script):

```bash
cd UniMatch-V2+CoVar

python main_unimatch_v2_covar.py \
  --config configs/unimatch_v2_covar_voc.yml \
  --labeled_id_path path/to/labeled_id.txt \
  --unlabeled_id_path path/to/unlabeled_id.txt \
  --val_id_path path/to/val_id.txt \
  --save_path path/to/save_dir
```

Again, training and evaluation follow the original UniMatch-V2 design, with CoVar taking care of pseudo-label reliability.

---

## Logging and Visualization

During training, we use `TensorBoardLogger` to record:

- training / validation losses
- evaluation metrics such as mIoU (if implemented in `SemiModule`)
- learning rate schedules
- other scalar metrics

To visualize the logs:

```bash
tensorboard --logdir path/to/save_dir/logs
```

Then open the displayed URL in your browser.

---

## License

This project is released under the **GNU General Public License v3.0 (GPL-3.0)**.  
See the [LICENSE](./license) file for the full text.

You are free to use, modify, and distribute this code under the terms of GPL-3.0.  
However, if you distribute software that is based on, or incorporates, this project, your software must also be released under GPL-3.0 and provide the corresponding source code.