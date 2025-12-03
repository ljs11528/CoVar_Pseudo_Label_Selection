#!/usr/bin/env python3
"""
Plot mean max-confidence (y) vs mean residual-variance (x) per sample.

Uses the PCOS metric from util/PCOS.py (get_max_confidence_and_residual_variance) to
compute per-pixel max-confidence and scaled residual variance, averages them over
valid pixels per image, and plots the scatter for the provided validation list.

Coloring: majority classes -> red, minority -> blue (majority determined the same
way as in new_test_majority.py by class pixel counts on the selected sample list).
Selected pseudo-labels (if simulation params provided) are outlined with a circle.

Defaults are chosen to match the workspace/requests: uses the 1449-sample val list
under `results_majority128/val_sample_1449.txt` if available and the provided ckpt
path can be overridden via CLI.
"""
import argparse
import os
import yaml
import random
import csv
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

from model.model_helper import ModelBuilder
from dataset.semi import SemiDataset
from util.classes import CLASSES
from util.PCOS import get_max_confidence_and_residual_variance


def read_val_list(path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def build_class_pixel_counts(val_lines, dataset_root, K):
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
    return counts


def determine_majority_classes(counts):
    K = counts.shape[0]
    order = np.argsort(-counts)
    top_n = int(np.ceil(K * 0.4))
    if top_n < 1:
        top_n = 1
    top_half = order[:top_n]
    majority_mask = np.zeros(K, dtype=bool)
    majority_mask[top_half] = True
    return majority_mask


def compute_per_image_resvar_and_conf(model, dataloader, device, K, majority_mask):
    model.to(device).eval()
    results = []
    for idx, batch in enumerate(dataloader):
        img, mask, id = batch
        bsize = img.shape[0]
        img = img.to(device)
        with torch.no_grad():
            out = model(img, False)
        if isinstance(out, (tuple, list)):
            out = out[0]
        probs = out.softmax(dim=1)

        for b in range(bsize):
            p = probs[b].cpu()
            m = mask[b].numpy()
            valid_mask = (m != 255)
            if valid_mask.sum() == 0:
                mean_max_conf = float('nan')
                mean_resvar = float('nan')
                dominant = -1
                group = 'unknown'
            else:
                # convert to tensors on cpu
                p_cpu = p
                vm = torch.from_numpy(valid_mask).to(dtype=torch.bool)
                # p_cpu: C x H x W ; need n x c x h x w
                p_batch = p_cpu.unsqueeze(0)
                vm_batch = vm.unsqueeze(0)
                max_confidence, scaled_residual_variance = get_max_confidence_and_residual_variance(
                    p_batch, vm_batch, K
                )
                # results shapes: [1, H, W]
                max_conf = max_confidence[0]
                resvar = scaled_residual_variance[0]
                # mask by valid_mask
                try:
                    vals = max_conf[vm]
                    mean_max_conf = float(vals.mean().item())
                except Exception:
                    mean_max_conf = float('nan')
                try:
                    vals2 = resvar[vm]
                    mean_resvar = float(vals2.mean().item())
                except Exception:
                    mean_resvar = float('nan')

                # dominant class
                vals_unique, counts = np.unique(m[valid_mask], return_counts=True)
                valid_idxs = (vals_unique >= 0) & (vals_unique < K)
                if valid_idxs.sum() == 0:
                    dominant = -1
                    group = 'unknown'
                else:
                    vals_u = vals_unique[valid_idxs]
                    counts_u = counts[valid_idxs]
                    dominant = int(vals_u[np.argmax(counts_u)])
                    group = 'majority' if majority_mask[dominant] else 'minority'

            rid = id[b] if isinstance(id[b], str) else id[b][0] if isinstance(id[b], (list, tuple)) else str(id[b])
            results.append({'id': rid, 'mean_max_conf': mean_max_conf, 'mean_resvar': mean_resvar, 'dominant_class': dominant, 'group': group, 'selected': False})

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1} / {len(dataloader)} batches")

    return results


def simulate_selection(results, selection_batch_size, select_top_k):
    m = int(selection_batch_size)
    k = int(select_top_k)
    for start in range(0, len(results), m):
        chunk = results[start: start + m]
        confs = np.array([c['mean_max_conf'] for c in chunk], dtype=np.float64)
        valid_mask = ~np.isnan(confs)
        valid_indices = np.where(valid_mask)[0]
        if valid_indices.size == 0:
            continue
        kk = min(k, valid_indices.size)
        sorted_idx = valid_indices[np.argsort(-confs[valid_indices])]
        chosen = sorted_idx[:kk]
        for ci in chosen:
            chunk[ci]['selected'] = True


def save_csv(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'per_image_conf_resvar.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['id', 'mean_max_conf', 'mean_resvar', 'dominant_class', 'group', 'selected']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Saved CSV to {csv_path}")


def plot_resvar(results, out_dir):
    # Separate samples by group and selection status
    xs_non_sel_major = [r['mean_resvar'] for r in results if (r['group'] == 'majority' and not r.get('selected', False))]
    ys_non_sel_major = [r['mean_max_conf'] for r in results if (r['group'] == 'majority' and not r.get('selected', False))]
    xs_non_sel_minor = [r['mean_resvar'] for r in results if (r['group'] == 'minority' and not r.get('selected', False))]
    ys_non_sel_minor = [r['mean_max_conf'] for r in results if (r['group'] == 'minority' and not r.get('selected', False))]

    xs_sel_major = [r['mean_resvar'] for r in results if (r['group'] == 'majority' and r.get('selected', False))]
    ys_sel_major = [r['mean_max_conf'] for r in results if (r['group'] == 'majority' and r.get('selected', False))]
    xs_sel_minor = [r['mean_resvar'] for r in results if (r['group'] == 'minority' and r.get('selected', False))]
    ys_sel_minor = [r['mean_max_conf'] for r in results if (r['group'] == 'minority' and r.get('selected', False))]

    plt.figure(figsize=(8, 6))
    # plot unselected samples: majority=red, minority=blue
    if len(xs_non_sel_minor) > 0:
        plt.scatter(xs_non_sel_minor, ys_non_sel_minor, s=18, alpha=0.5, c='blue', label='Minority class(unselected)')
    if len(xs_non_sel_major) > 0:
        plt.scatter(xs_non_sel_major, ys_non_sel_major, s=18, alpha=0.5, c='red', label='Majority class(unselected)')

    # plot selected samples with distinct colors: majority=orange, minority=purple
    if len(xs_sel_major) > 0:
        plt.scatter(xs_sel_major, ys_sel_major, s=18, alpha=0.9, c='orange', label='Majority class(selected)')
    if len(xs_sel_minor) > 0:
        plt.scatter(xs_sel_minor, ys_sel_minor, s=18, alpha=0.9, c='purple', label='Minority class(selected)')

    plt.xlabel('Residual-class variance')
    plt.ylabel('Max-Confidence')
    plt.title('Per-sample Max-Confidence vs Residual-class variance')
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.5)
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, 'conf_vs_resvar.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved plot to {fig_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pascal.yaml')
    parser.add_argument('--ckpt', default='/workspace/coredata/CSL/exp/pascal/CSL/r101/1_4/checkpoints/epoch=54-val_mIOU=79.57.ckpt')
    parser.add_argument('--val_id_path', default='/workspace/coredata/CSL_new/results_majority128/val_sample_1449.txt')
    parser.add_argument('--out_dir', default='results_resvar')
    parser.add_argument('--selection_batch_size', type=int, default=None)
    parser.add_argument('--select_top_k', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # try to cd to project root similar to other scripts
    try:
        config_abspath = os.path.abspath(args.config)
        project_root = os.path.dirname(os.path.dirname(config_abspath))
        if os.path.isdir(project_root):
            os.chdir(project_root)
    except Exception:
        pass

    if not os.path.exists(args.val_id_path):
        # fallback: try building val ids like the other script
        dataset_root = cfg['dataset']['root']
        val_id_path = os.path.join(dataset_root, 'val_generated_ids.txt')
        try:
            from test import build_val_id_from_folders
            build_val_id_from_folders(dataset_root, out_path=val_id_path)
            args.val_id_path = val_id_path
        except Exception:
            raise FileNotFoundError(f"val id list not found at {args.val_id_path}")

    val_lines = read_val_list(args.val_id_path)
    dataset_root = cfg['dataset']['root']
    K = cfg.get('nclass') or cfg.get('train', {}).get('nclass') or cfg['model']['decoder']['kwargs'].get('nclass')

    print('Counting class pixels across selected validation samples...')
    counts = build_class_pixel_counts(val_lines, dataset_root, K)
    majority_mask = determine_majority_classes(counts)
    print('Class counts:', counts)
    print('Majority classes (indices):', np.where(majority_mask)[0].tolist())

    valset = SemiDataset(cfg['dataset']['name'], cfg['dataset']['root'], 'val', args.val_id_path, crop_size=cfg['dataset'].get('crop_size', None))
    from torch.utils.data import DataLoader
    valloader = DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True, shuffle=False)

    model = ModelBuilder(cfg['model'])
    # load checkpoint similar to original script
    ckpt = args.ckpt
    if ckpt is not None:
        loaded = torch.load(ckpt, map_location='cpu')
        if isinstance(loaded, dict) and 'state_dict' in loaded:
            state_dict = loaded['state_dict']
        else:
            state_dict = loaded
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

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    results = compute_per_image_resvar_and_conf(model, valloader, device, K, majority_mask)

    if args.selection_batch_size is not None and args.select_top_k is not None:
        simulate_selection(results, args.selection_batch_size, args.select_top_k)

    save_csv(results, args.out_dir)
    plot_resvar(results, args.out_dir)


if __name__ == '__main__':
    main()
