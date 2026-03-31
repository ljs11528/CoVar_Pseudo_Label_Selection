import argparse
import json
import os
import random

import numpy as np
import torch

from util.PCOS import get_max_confidence_and_residual_variance


class PseudoLabelArtifactsCollector:
    def __init__(self, save_path, num_classes, sample_size=100000, rng_seed=42):
        self.output_dir = os.path.join(save_path, 'pseudo_diagnostics')
        self.num_classes = num_classes
        self.sample_size = sample_size
        self._rng = random.Random(rng_seed)

        self.total_pseudo_pixels = 0
        self.total_selected_pixels = 0
        self.epoch_fractions = []
        self.epoch_class_stats = []
        self.scatter_epoch = None
        self._sample_seen_count = 0
        self._sample_rcv = []
        self._sample_mc = []
        self._sample_selected = []
        self._reset_epoch_state()

    def _reset_epoch_state(self):
        self.epoch_pseudo_pixels = 0
        self.epoch_selected_pixels = 0
        self.epoch_class_generated = [0] * self.num_classes
        self.epoch_class_selected = [0] * self.num_classes

    @torch.no_grad()
    def update_batch(self, pred_logits, ignore_mask, conf_mask, pseudo_mask, current_epoch):
        valid_mask = ignore_mask != 255
        batch_pseudo = int(valid_mask.sum().item())
        if batch_pseudo == 0:
            return

        selected_mask = conf_mask & valid_mask
        batch_selected = int(selected_mask.sum().item())

        self.total_pseudo_pixels += batch_pseudo
        self.total_selected_pixels += batch_selected
        self.epoch_pseudo_pixels += batch_pseudo
        self.epoch_selected_pixels += batch_selected

        labels = pseudo_mask[valid_mask].detach().cpu()
        if labels.numel() > 0:
            generated = torch.bincount(labels, minlength=self.num_classes).tolist()
            for class_index, count in enumerate(generated):
                self.epoch_class_generated[class_index] += int(count)

        selected_labels = pseudo_mask[selected_mask].detach().cpu()
        if selected_labels.numel() > 0:
            selected = torch.bincount(selected_labels, minlength=self.num_classes).tolist()
            for class_index, count in enumerate(selected):
                self.epoch_class_selected[class_index] += int(count)

        probs = pred_logits.softmax(dim=1)
        max_confidence, scaled_residual_variance = get_max_confidence_and_residual_variance(
            probs,
            valid_mask,
            self.num_classes,
        )
        mc_vals = max_confidence[valid_mask].detach().cpu().numpy().ravel()
        rcv_vals = scaled_residual_variance[valid_mask].detach().cpu().numpy().ravel()
        sel_vals = selected_mask[valid_mask].detach().cpu().numpy().astype(np.uint8).ravel()

        if mc_vals.size == 0:
            return

        if self.scatter_epoch is None:
            self.scatter_epoch = int(current_epoch)

        sample_count = min(mc_vals.shape[0], self.sample_size)
        if mc_vals.shape[0] > sample_count:
            chosen_indices = self._rng.sample(range(mc_vals.shape[0]), sample_count)
        else:
            chosen_indices = range(mc_vals.shape[0])

        for idx in chosen_indices:
            self._sample_seen_count += 1
            if len(self._sample_rcv) < self.sample_size:
                self._sample_rcv.append(float(rcv_vals[idx]))
                self._sample_mc.append(float(mc_vals[idx]))
                self._sample_selected.append(int(sel_vals[idx]))
                continue

            replace_at = self._rng.randint(1, self._sample_seen_count)
            if replace_at <= self.sample_size:
                replace_pos = replace_at - 1
                self._sample_rcv[replace_pos] = float(rcv_vals[idx])
                self._sample_mc[replace_pos] = float(mc_vals[idx])
                self._sample_selected[replace_pos] = int(sel_vals[idx])

    def on_epoch_end(self, current_epoch):
        fraction = 0.0
        if self.epoch_pseudo_pixels > 0:
            fraction = self.epoch_selected_pixels / self.epoch_pseudo_pixels

        self.epoch_fractions.append({
            'epoch': int(current_epoch),
            'generated_pixels': int(self.epoch_pseudo_pixels),
            'selected_pixels': int(self.epoch_selected_pixels),
            'selected_fraction': float(fraction),
        })
        self.epoch_class_stats.append({
            'epoch': int(current_epoch),
            'generated_per_class': list(self.epoch_class_generated),
            'selected_per_class': list(self.epoch_class_selected),
        })
        self._reset_epoch_state()

    def write_artifacts(self, logger=None):
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            'total_generated_pixels': int(self.total_pseudo_pixels),
            'total_selected_pixels': int(self.total_selected_pixels),
            'selected_fraction': float(self.total_selected_pixels / self.total_pseudo_pixels) if self.total_pseudo_pixels else 0.0,
            'scatter_epoch': self.scatter_epoch,
            'sample_size': int(self.sample_size),
            'epochs': self.epoch_fractions,
            'class_stats': self.epoch_class_stats,
        }
        summary_path = os.path.join(self.output_dir, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)

        scatter_path = None
        if self._sample_rcv:
            scatter_path = os.path.join(self.output_dir, 'scatter_sample.npz')
            np.savez_compressed(
                scatter_path,
                rcv=np.asarray(self._sample_rcv, dtype=np.float32),
                mc=np.asarray(self._sample_mc, dtype=np.float32),
                selected=np.asarray(self._sample_selected, dtype=np.uint8),
            )

        message = f'Wrote pseudo-label artifacts to {self.output_dir}'
        if scatter_path is not None:
            message += f' (scatter sample: {scatter_path})'

        if logger is not None:
            logger.info(message)
        else:
            print(message)


def render_artifacts(input_dir, output_dir=None):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = output_dir or input_dir
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(input_dir, 'summary.json')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f'Missing summary file: {summary_path}')

    with open(summary_path, 'r', encoding='utf-8') as handle:
        summary = json.load(handle)

    written_files = []

    epochs = summary.get('epochs', [])
    if epochs:
        epoch_ids = [int(item['epoch']) for item in epochs]
        fractions = [float(item['selected_fraction']) for item in epochs]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epoch_ids, fractions, marker='o', linestyle='-')
        ax.set_xlabel('epoch')
        ax.set_ylabel('selected / generated')
        ax.set_title('Pseudo-label selection fraction per epoch')
        ax.grid(True)
        fraction_path = os.path.join(output_dir, 'pseudo_fraction_final.png')
        fig.tight_layout()
        fig.savefig(fraction_path)
        plt.close(fig)
        written_files.append(fraction_path)

    scatter_path = os.path.join(input_dir, 'scatter_sample.npz')
    if os.path.exists(scatter_path):
        scatter_data = np.load(scatter_path)
        rcv_vals = scatter_data['rcv']
        mc_vals = scatter_data['mc']
        selected_vals = scatter_data['selected']
        scatter_epoch = summary.get('scatter_epoch')

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(rcv_vals[selected_vals == 0], mc_vals[selected_vals == 0], s=6, c='blue', label='unselected', alpha=0.6)
        if np.any(selected_vals == 1):
            ax.scatter(rcv_vals[selected_vals == 1], mc_vals[selected_vals == 1], s=6, c='red', label='selected', alpha=0.6)
        ax.set_xlabel('RCV (residual class variance)')
        ax.set_ylabel('MC (max confidence)')
        if scatter_epoch is None:
            ax.set_title('Pseudo samples scatter')
            scatter_png = os.path.join(output_dir, 'pseudo_scatter.png')
        else:
            ax.set_title(f'Pseudo samples scatter (epoch={scatter_epoch})')
            scatter_png = os.path.join(output_dir, f'pseudo_scatter_epoch_{scatter_epoch}.png')
        ax.legend(markerscale=3)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(scatter_png)
        plt.close(fig)
        written_files.append(scatter_png)

    return written_files


def main():
    parser = argparse.ArgumentParser(description='Render pseudo-label diagnostic artifacts produced during CSL training.')
    parser.add_argument('--input_dir', required=True, help='Directory containing summary.json and scatter_sample.npz')
    parser.add_argument('--output_dir', default=None, help='Directory to place generated plots; defaults to input_dir')
    args = parser.parse_args()

    written_files = render_artifacts(args.input_dir, args.output_dir)
    if written_files:
        for path in written_files:
            print(path)
    else:
        print('No plots were generated.')


if __name__ == '__main__':
    main()