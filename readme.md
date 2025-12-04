# CoVar: A Confidence–Variance Theory for Pseudo-Label Selection in Semi-Supervised Learning

This repository contains the implementation of **CoVar**, a confidence–variance based theory for pseudo-label selection in semi-supervised learning.  
CoVar models covariate consistency and prediction uncertainty to assess the reliability of pseudo-labels and selectively use them during training, improving robustness and accuracy under limited annotations.

We apply CoVar to three representative semi-supervised methods:

- `CSL+CoVar/`: CoVar integrated into the **CSL** framework for semi-supervised semantic segmentation  
- `SimPLE+CoVar/`: CoVar integrated into **SimPLE** (Similar Pseudo Label Exploitation) for semi-supervised classification  
- `UniMatch-V2+CoVar/`: CoVar integrated into **UniMatch-V2** for semi-supervised semantic segmentation  

Each subdirectory contains a self-contained implementation with its own configuration and (in some cases) original upstream README for reference.

---

## Repository Structure

- `CSL+CoVar/`  
  Semi-supervised **segmentation** (e.g., PASCAL VOC / Cityscapes) with CoVar-integrated CSL.  
  Main files:
  - `CSL.py`: training entry point for CSL+CoVar
  - `dataset/`, `model/`, `train/`, `configs/`: dataset, model, training logic, and experiment configs

- `SimPLE+CoVar/`  
  Semi-supervised **classification** with CoVar integrated into **SimPLE**.  
  This directory is based on the official SimPLE implementation:
  > Hu et al., *SimPLE: Similar Pseudo Label Exploitation for Semi-Supervised Classification*, CVPR 2021  
  We keep the original SimPLE README and scripts (e.g., `main.py`, `main_ddp.py`, `runs/*.txt`) and add CoVar-related components on top.

- `UniMatch-V2+CoVar/`  
  Semi-supervised **segmentation** with CoVar integrated into UniMatch-V2.  
  Similar structure to CSL+CoVar, with its own configs and training script.

Below we highlight how to run each subproject, and how CoVar fits into the pipeline.

---

## 1. Environment

We recommend Python 3.8+.

At the top level, you may start with:

```bash
conda create -n covar python=3.8 -y
conda activate covar

pip install -r requirements.txt
# If there is no global requirements.txt, please install dependencies
# according to each subdirectory (CSL+CoVar, SimPLE+CoVar, UniMatch-V2+CoVar).
```

### 1.1 CSL+CoVar / UniMatch-V2+CoVar (Segmentation)

Typical dependencies:

- PyTorch + torchvision (GPU version matched to your CUDA / driver)
- pytorch-lightning
- pyyaml
- numpy, etc.

If a segmentation-specific `requirements.txt` or `environment.yaml` is provided under `CSL+CoVar/` or `UniMatch-V2+CoVar/`, please follow those.

### 1.2 SimPLE+CoVar (Classification)

Inside `SimPLE+CoVar/`, we essentially keep the original SimPLE environment:

- Python 3.6 or newer
- [PyTorch](https://pytorch.org/) 1.6.0 or newer
- [torchvision](https://pytorch.org/docs/stable/torchvision/index.html) 0.7.0 or newer
- [kornia](https://kornia.readthedocs.io/en/latest/augmentation.html) 0.5.0 or newer
- numpy
- scikit-learn
- [plotly](https://plotly.com/python/) 4.0.0 or newer
- wandb 0.9.0 or newer (optional; for logging to Weights & Biases)

**Recommended versions (from original SimPLE):**

| Python | PyTorch | torchvision | kornia |
| ------ | ------- | ---------- | ------ |
| 3.8.5  | 1.6.0   | 0.7.0      | 0.5.0  |

Install via `pip`:

```bash
cd SimPLE+CoVar
pip install -r requirements.txt
```

or via `conda` (if `environment.yaml` is provided):

```bash
cd SimPLE+CoVar
conda env create -f environment.yaml
```

---

## 2. Data Preparation

### 2.1 CSL+CoVar (Semantic Segmentation)

Dataset roots and split files are specified in YAML config files under `CSL+CoVar/configs/`.

Example directory layout (PASCAL VOC style):

```text
CSL+CoVar/
  data/
    Pascal/
      JPEGImages/
      SegmentationClass/
      ...
```

You also need text files listing image IDs:

- `labeled_id.txt` – IDs for labeled samples
- `unlabeled_id.txt` – IDs for unlabeled samples
- `val_id.txt` – IDs for validation samples

These are passed to `CSL.py` via command-line arguments.

### 2.2 UniMatch-V2+CoVar (Semantic Segmentation)

Similar to CSL+CoVar: please check the configs under `UniMatch-V2+CoVar/configs/` for dataset root, splits, and data format.

### 2.3 SimPLE+CoVar (Classification)

SimPLE (and SimPLE+CoVar) supports multiple benchmarks such as CIFAR-10/100, SVHN and Mini-ImageNet.

Data-related arguments are specified via text arg files under `SimPLE+CoVar/runs/`, for example:

- `runs/cifar10_args.txt`
- `runs/cifar100_args.txt`
- `runs/miniimagenet_args.txt`

These files contain command-line flags for `main.py` / `main_ddp.py`, including dataset path, number of labeled samples, etc.  
Please refer to the original `SimPLE+CoVar/README.md` for options such as `data_root`, `num-labeled`, `batch-size`, etc.

---

## 3. Training

### 3.1 CSL+CoVar

From repository root:

```bash
cd CSL+CoVar

python CSL.py \
  --config configs/csl_covar_voc.yml \
  --labeled_id_path path/to/labeled_id.txt \
  --unlabeled_id_path path/to/unlabeled_id.txt \
  --val_id_path path/to/val_id.txt \
  --save_path path/to/save_dir
```

- `--config`: YAML config (model, optimizer, dataset root, etc.)
- `--labeled_id_path`: labeled IDs file
- `--unlabeled_id_path`: unlabeled IDs file
- `--val_id_path`: validation IDs file
- `--save_path`: output directory (logs, checkpoints)

`CSL.py` builds the model via `ModelBuilder`, constructs three `DataLoader`s (`labeled`, `unlabeled`, `val`) using `SemiDataset`, and runs training with `pytorch_lightning.Trainer`.  
CoVar’s pseudo-label selection logic is integrated inside the training module (e.g., `SemiModule` and related components) to filter or re-weight pseudo-labels.

For multi-GPU / mixed precision, the Trainer is typically configured with:

- `accelerator="gpu"`, DDP strategy
- `precision="bf16-mixed"` or `16-mixed` depending on your hardware
- automatic checkpointing under `--save_path/checkpoints`
- TensorBoard logs under `--save_path/logs`

You can control DataLoader behaviour using environment variables:

```bash
export NUM_WORKERS=4
export PIN_MEMORY=1
```

### 3.2 UniMatch-V2+CoVar

From repository root:

```bash

sh scripts/train.sh <num_gpu> <port>
# to fully reproduce our results, the <num_gpu> should be set as 4 on all four datasets
# otherwise, you need to adjust the learning rate accordingly

# or use slurm
# sh scripts/slurm_train.sh <num_gpu> <port> <partition>
```

To train on other datasets or splits, please modify
``dataset`` and ``split`` in [train.sh](https://github.com/LiheYoung/UniMatch-V2/blob/main/scripts/train.sh).

### FixMatch

Modify the ``method`` from ``'unimatch_v2'`` to ``'fixmatch'`` in [train.sh](https://github.com/LiheYoung/UniMatch-V2/blob/main/scripts/train.sh).

### Supervised Baseline

Modify the ``method`` from ``'unimatch_v2'`` to ``'supervised'`` in [train.sh](https://github.com/LiheYoung/UniMatch-V2/blob/main/scripts/train.sh).
```

(Please adapt script name and config file to match your actual implementation.)

The usage is analogous to CSL+CoVar, with UniMatch-V2’s training logic extended by CoVar’s pseudo-label selection.

### 3.3 SimPLE+CoVar

From repository root:

```bash
cd SimPLE+CoVar
```

#### 3.3.1 Single-GPU Example (Mini-ImageNet)

To reproduce Mini-ImageNet-style results with SimPLE+CoVar:

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="0" \
python main.py \
@runs/miniimagenet_args.txt
```

#### 3.3.2 Single-GPU Example (CIFAR-10)

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="0" \
python main.py \
@runs/cifar10_args.txt
```

#### 3.3.3 Multi-GPU Example (CIFAR-100, DDP)

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="0,1" \
python -m torch.distributed.launch \
  --nproc_per_node=2 main_ddp.py \
  @runs/cifar100_args.txt \
  --num-epochs 2048 \
  --num-step-per-epoch 512
```

Here:

- `main.py` / `main_ddp.py` are the main SimPLE training scripts
- `@runs/*.txt` includes arguments such as dataset, architecture, number of labeled samples, etc.
- CoVar is integrated into the pseudo-label generation / selection part of the training loop, re-using SimPLE’s high-confidence pseudo-label exploitation framework.

For more details on specific flags, please see `SimPLE+CoVar/README.md` and the corresponding `runs/*.txt`.

---

## 4. CoVar: Pseudo-Label Selection (Conceptual Overview)

Across the three methods (CSL, SimPLE, UniMatch-V2), CoVar follows the same high-level idea:

1. Generate pseudo-labels on unlabeled data (using the current model, EMA / teacher model, or consistency-based method).
2. For each pixel (segmentation) or sample / feature (classification), compute covariate statistics:
   - prediction confidence
   - distributional properties (e.g., class probabilities, margins)
   - consistency across augmentations / views
   - variance / uncertainty measures
3. Use a confidence–variance based criterion to:
   - filter out low-reliability pseudo-labels, or
   - down-weight them in the training loss
4. Plug this selection mechanism into the base method:
   - CSL+CoVar: applies CoVar on per-pixel pseudo-labels in segmentation
   - SimPLE+CoVar: applies CoVar on high-confidence pseudo-labels exploited by SimPLE
   - UniMatch-V2+CoVar: applies CoVar on pseudo-labels in the UniMatch-V2 consistency framework

This design allows CoVar to be flexibly combined with diverse semi-supervised learning pipelines while enforcing a “reliable pseudo-label first” principle.

---

## 5. Logging and Visualization

For segmentation projects (CSL+CoVar, UniMatch-V2+CoVar), we use `TensorBoardLogger` from PyTorch Lightning:

```bash
tensorboard --logdir path/to/save_dir/logs
```

For SimPLE+CoVar, you can additionally enable Weights & Biases (wandb) logging if desired (see `SimPLE+CoVar/utils/loggers.py` and the original SimPLE README).

---

## 6. Citation

If you use this repository or CoVar in your research, please consider citing:

```bibtex

```

For CSL, SimPLE, Unimatch V2 please also cite:

```bibtex
@InProceedings{Liu_2025_ICCV,
    author    = {Liu, Pan and Liu, Jinshi},
    title     = {When Confidence Fails: Revisiting Pseudo-Label Selection in Semi-supervised Semantic Segmentation},
    booktitle = {ICCV},
    month     = {October},
    year      = {2025},
    pages     = {21874-21884}
}

@InProceedings{Hu-2020-SimPLE,
  author = {{Hu*}, Zijian and {Yang*}, Zhengyu and Hu, Xuefeng and Nevaita, Ram},
  title = {{SimPLE}: {S}imilar {P}seudo {L}abel {E}xploitation for {S}emi-{S}upervised {C}lassification},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2021},
  url = {https://arxiv.org/abs/2103.16725},
}

@article{Unimatchv2,
  title={Unimatch v2: Pushing the limit of semi-supervised semantic segmentation},
  author={Yang, Lihe and Zhao, Zhen and Zhao, Hengshuang},
  journal={IEEE TPAMI},
  year={2025},
  publisher={IEEE}
}
```

---

## 7. License

This project is released under the **GNU General Public License v3.0 (GPL-3.0)**.  
See the [LICENSE](./license) file for the full text.

You are free to use, modify, and distribute this code under the terms of GPL-3.0.  
However, if you distribute software that is based on, or incorporates, this project, your software must also be released under GPL-3.0 and provide the corresponding source code.