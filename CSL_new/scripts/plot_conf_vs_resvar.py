#!/usr/bin/env python3
"""
Plot per-sample Max-Confidence (y) vs Residual-Variance (x) using PCOS metrics.

This script processes the validation set used by `new_test_majority.py` (default val list
from config) and computes for each sample:
  - mean(max_confidence) over valid pixels
  - mean(scaled_residual_variance) over valid pixels

Samples are colored red (majority) or blue (minority) according to the same class-splitting
used in `new_test_majority.py`. Samples that have any selected pseudo-label pixels (as defined
by the confident mask used in SemiModule.get_weight) are plotted with a star marker, others
with a circle.

Note: selection criterion = fraction of valid pixels with confident_mask > 0 (i.e., selected
pixels) > 0 (any selected pixels). This matches the `mask_ratio` diagnostic from training.

Usage:
  python scripts/plot_conf_vs_resvar.py --config configs/pascal.yaml --ckpt <ckpt_path> --out_dir results

"""
import argparse
import os
import sys
import yaml
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path so imports like `import model` work when the
# script is executed from any working directory.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.model_helper import ModelBuilder
from dataset.semi import SemiDataset
from util.PCOS import get_max_confidence_and_residual_variance, batch_class_stats


def load_ckpt_to_model(model, ckpt_path, map_location='cpu'):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith('model.'):
            new_k = new_k[len('model.'):]
        if new_k.startswith('module.'):
            new_k = new_k[len('module.'):]
        if new_k.startswith('model.'):
            new_k = new_k[len('model.'):]
        new_state[new_k] = v

    model.load_state_dict(new_state, strict=False)


def read_val_list(path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def determine_majority_classes_from_counts(val_lines, dataset_root, K):
    # reuse logic from new_test_majority: count pixels per class
    counts = np.zeros(K, dtype=np.int64)
    for line in val_lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        mask_rel = parts[1]
        mask_path = os.path.join(dataset_root, mask_rel)
        if not os.path.exists(mask_path):
            continue
        mask = np.array(Image.open(mask_path))
        mask = mask.astype(np.int64)
        mask = mask[mask != 255]
        if mask.size == 0:
            continue
        mask = mask[(mask >= 0) & (mask < K)]
        if mask.size == 0:
            continue
        hist = np.bincount(mask, minlength=K)
        counts += hist
    # pick top 40% as majority (same as new_test_majority)
    order = np.argsort(-counts)
    top_n = int(np.ceil(K * 0.4))
    top_n = max(1, top_n)
    majority_mask = np.zeros(K, dtype=bool)
    majority_mask[order[:top_n]] = True
    return majority_mask, counts


def compute_weight_from_pcos(pred, valid_mask, num_classes, epsilon=1e-8, alpha=2.0):
    # pred: Tensor [1, C, H, W] (softmax probabilities)
    # valid_mask: Bool Tensor [1, H, W]
    max_confidence, scaled_residual_variance = get_max_confidence_and_residual_variance(pred, valid_mask, num_classes, epsilon)
    means, vars = batch_class_stats(max_confidence, scaled_residual_variance, num_classes)
    conf_mean = means[:, 0].view(-1, 1, 1)
    res_mean = means[:, 1].view(-1, 1, 1)
    conf_var = vars[:, 0].view(-1, 1, 1)
    res_var = vars[:, 1].view(-1, 1, 1)

    conf_z = (max_confidence - conf_mean) / torch.sqrt(conf_var + epsilon)
    res_z = (res_mean - scaled_residual_variance) / torch.sqrt(res_var + epsilon)

    weight_conf = torch.exp(- (conf_z ** 2) / alpha)
    weight_res = torch.exp(- (res_z ** 2) / alpha)
    weight = weight_conf * weight_res
    confident_mask = (conf_z > 0) | (res_z > 0)
    # where confident_mask true => weight set to 1
    weight = torch.where(confident_mask, torch.ones_like(weight), weight)
    # final mask of selected pseudo-label pixels (1 where confident and valid)
    selected_pixels = confident_mask & valid_mask
    return max_confidence, scaled_residual_variance, selected_pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=False, help='checkpoint to load (optional)')
    parser.add_argument('--val_id_path', required=False)
    parser.add_argument('--out_dir', default='results_conf_vs_resvar')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    try:
        config_abspath = os.path.abspath(args.config)
        project_root = os.path.dirname(os.path.dirname(config_abspath))
        if os.path.isdir(project_root):
            os.chdir(project_root)
    except Exception:
        pass

    test_cfg = cfg.get('test', {})
    val_id_path = args.val_id_path or test_cfg.get('val_id_path')
    if val_id_path is None:
        dataset_root = cfg['dataset']['root']
        val_id_path = os.path.join(dataset_root, 'val_generated_ids.txt')
        from test import build_val_id_from_folders
        build_val_id_from_folders(dataset_root, out_path=val_id_path)

    val_lines = read_val_list(val_id_path)
    # expect 1449 samples as in new_test_majority.py
    print(f'Read {len(val_lines)} val lines from {val_id_path}')

    dataset_root = cfg['dataset']['root']
    K = cfg.get('nclass') or cfg.get('train', {}).get('nclass') or cfg['model']['decoder']['kwargs'].get('nclass')

    majority_mask, counts = determine_majority_classes_from_counts(val_lines, dataset_root, K)
    print('Computed class counts and majority mask')

    # build val dataset and loader (batch_size=1)
    valset = SemiDataset(cfg['dataset']['name'], dataset_root, 'val', val_id_path)
    from torch.utils.data import DataLoader
    valloader = DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True, shuffle=False)

    # load model if provided
    model = None
    if args.ckpt:
        model = ModelBuilder(cfg['model'])
        load_ckpt_to_model(model, args.ckpt, map_location='cpu')
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    results = []
    # iterate valloader
    for idx, batch in enumerate(valloader):
        img, mask, sid = batch
        img = img.to(device)
        # run model if available to get predicted probabilities; otherwise skip
        if model is None:
            print('No model provided; exiting')
            break
        model.to(device).eval()
        with torch.no_grad():
            out = model(img, False)
        if isinstance(out, (tuple, list)):
            out = out[0]
        probs = out.softmax(dim=1)

        # prepare tensors for PCOS: preds probabilities and valid mask
        probs_cpu = probs.cpu()
        mask_cpu = mask
        valid_mask = (mask_cpu != 255)

        # call PCOS routines
        max_conf, scaled_res_var = get_max_confidence_and_residual_variance(probs_cpu, valid_mask, K)
        # max_conf, scaled_res_var are [1, H, W]
        # compute per-image means over valid pixels
        vm = valid_mask[0]
        if vm.sum() == 0:
            mean_max_conf = float('nan')
            mean_res_var = float('nan')
            frac_selected = 0.0
        else:
            mean_max_conf = float(max_conf[0][vm].mean().item())
            mean_res_var = float(scaled_res_var[0][vm].mean().item())
            # compute selected pixels using same logic as SemiModule.get_weight
            max_conf_t = max_conf
            scaled_res_t = scaled_res_var
            means, vars = batch_class_stats(max_conf_t, scaled_res_t, K)
            conf_mean = means[:, 0].view(-1, 1, 1)
            res_mean = means[:, 1].view(-1, 1, 1)
            conf_var = vars[:, 0].view(-1, 1, 1)
            res_var = vars[:, 1].view(-1, 1, 1)
            conf_z = (max_conf_t - conf_mean) / torch.sqrt(conf_var + 1e-8)
            res_z = (res_mean - scaled_res_t) / torch.sqrt(res_var + 1e-8)
            confident_mask = (conf_z > 0) | (res_z > 0)
            selected_pixels = (confident_mask & valid_mask)
            frac_selected = float(selected_pixels[0].sum().item()) / float(vm.sum().item())

        # determine dominant class and majority/minority group
        sid_str = sid[0] if isinstance(sid[0], str) else str(sid[0])
        m = mask_cpu[0].numpy()
        vm_np = vm.numpy()
        if vm_np.sum() == 0:
            dominant = -1
            group = 'unknown'
        else:
            vals, cnts = np.unique(m[vm_np], return_counts=True)
            vals = vals[(vals >= 0) & (vals < K)]
            if vals.size == 0:
                dominant = -1
                group = 'unknown'
            else:
                # compute counts over valid pixels
                all_vals, all_cnts = np.unique(m[vm_np], return_counts=True)
                # select the value with max count
                dominant = int(all_vals[np.argmax(all_cnts)])
                group = 'majority' if majority_mask[dominant] else 'minority'

        selected_flag = frac_selected > 0.0

        results.append({
            'id': sid_str,
            'mean_max_conf': mean_max_conf,
            'mean_res_var': mean_res_var,
            'dominant_class': int(dominant) if dominant is not None else -1,
            'group': group,
            'frac_selected': frac_selected,
            'selected': bool(selected_flag)
        })

        if (idx + 1) % 100 == 0:
            print(f'Processed {idx+1} / {len(valloader)}')

    # plotting
    os.makedirs(args.out_dir, exist_ok=True)
    # build masks
    groups = [r['group'] for r in results]
    xs = np.array([r['mean_res_var'] for r in results], dtype=np.float64)
    ys = np.array([r['mean_max_conf'] for r in results], dtype=np.float64)
    sel = np.array([r['selected'] for r in results], dtype=bool)
    maj = np.array([1 if g == 'majority' else 0 for g in groups], dtype=bool)

    maj_sel = maj & sel
    maj_nsel = maj & (~sel)
    min_sel = (~maj) & sel
    min_nsel = (~maj) & (~sel)

    plt.figure(figsize=(8, 6))
    # plot majority not selected: red circles
    plt.scatter(xs[maj_nsel], ys[maj_nsel], c='red', marker='o', label='majority (not selected)', alpha=0.7, s=30)
    # majority selected: red stars
    plt.scatter(xs[maj_sel], ys[maj_sel], c='red', marker='*', label='majority (selected)', s=80)
    # minority not selected: blue circles
    plt.scatter(xs[min_nsel], ys[min_nsel], c='blue', marker='o', label='minority (not selected)', alpha=0.7, s=30)
    # minority selected: blue stars
    plt.scatter(xs[min_sel], ys[min_sel], c='blue', marker='*', label='minority (selected)', s=80)

    plt.xlabel('Scaled Residual Variance (PCOS)')
    plt.ylabel('Mean Max-Confidence')
    plt.title('Mean Max-Confidence vs Scaled Residual Variance')
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.5)
    fig_path = os.path.join(args.out_dir, 'conf_vs_resvar.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f'Saved plot to {fig_path}')

    # save CSV
    import csv
    csv_path = os.path.join(args.out_dir, 'per_image_conf_resvar.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'mean_max_conf', 'mean_res_var', 'dominant_class', 'group', 'frac_selected', 'selected'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f'Saved CSV to {csv_path}')


if __name__ == '__main__':
    main()
