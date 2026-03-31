import json
import os

import torch


class PseudoLabelMetricsTracker:
    def __init__(self, save_path):
        self.save_path = save_path
        self.generated_pixels = 0
        self.selected_pixels = 0
        self.epoch_generated_pixels = 0
        self.epoch_selected_pixels = 0
        self.cumulative_generated_pixels = 0
        self.cumulative_selected_pixels = 0
        self.epoch_history = []

    def update_batch(self, ignore_mask, confidence_mask):
        valid_mask = ignore_mask != 255
        generated = int(valid_mask.sum().item())
        selected = int((valid_mask & confidence_mask).sum().item())
        self.generated_pixels += generated
        self.selected_pixels += selected
        self.epoch_generated_pixels += generated
        self.epoch_selected_pixels += selected

    def aggregate_distributed_pair(self, generated, selected, device):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            stats_tensor = torch.tensor([generated, selected], dtype=torch.long, device=device)
            torch.distributed.all_reduce(stats_tensor, op=torch.distributed.ReduceOp.SUM)
            generated = int(stats_tensor[0].item())
            selected = int(stats_tensor[1].item())
        return generated, selected

    def aggregate_distributed(self, device):
        return self.aggregate_distributed_pair(self.generated_pixels, self.selected_pixels, device)

    def on_epoch_end(self, current_epoch, trainer, module, logger=None):
        device = module.device if isinstance(module.device, torch.device) else torch.device('cpu')
        epoch_generated, epoch_selected = self.aggregate_distributed_pair(
            self.epoch_generated_pixels,
            self.epoch_selected_pixels,
            device,
        )

        is_global_zero = getattr(trainer, 'is_global_zero', True)
        if is_global_zero:
            self.cumulative_generated_pixels += epoch_generated
            self.cumulative_selected_pixels += epoch_selected
            epoch_selection_rate = float(epoch_selected / epoch_generated) if epoch_generated > 0 else 0.0
            cumulative_selection_rate = (
                float(self.cumulative_selected_pixels / self.cumulative_generated_pixels)
                if self.cumulative_generated_pixels > 0
                else 0.0
            )

            record = {
                'epoch': int(current_epoch),
                'generated_pseudo_labels_epoch': int(epoch_generated),
                'selected_pseudo_labels_epoch': int(epoch_selected),
                'selection_rate_epoch': float(epoch_selection_rate),
                'generated_pseudo_labels_cumulative': int(self.cumulative_generated_pixels),
                'selected_pseudo_labels_cumulative': int(self.cumulative_selected_pixels),
                'selection_rate_cumulative': float(cumulative_selection_rate),
            }
            self.epoch_history.append(record)

            if logger is not None:
                logger.info(
                    'Pseudo-label epoch summary | '
                    f"epoch={record['epoch']}, "
                    f"generated_epoch={record['generated_pseudo_labels_epoch']}, "
                    f"selected_epoch={record['selected_pseudo_labels_epoch']}, "
                    f"epoch_rate={record['selection_rate_epoch']:.6f}, "
                    f"generated_cum={record['generated_pseudo_labels_cumulative']}, "
                    f"selected_cum={record['selected_pseudo_labels_cumulative']}, "
                    f"cum_rate={record['selection_rate_cumulative']:.6f}"
                )

        self.epoch_generated_pixels = 0
        self.epoch_selected_pixels = 0

    @torch.no_grad()
    def evaluate_pseudo_accuracy(self, module, val_loader):
        if val_loader is None:
            return None

        was_training = module.training
        module.eval()

        total_valid = 0
        total_correct = 0

        for batch in val_loader:
            img, mask, _ = batch
            img = img.to(module.device, non_blocking=True)
            mask = mask.to(module.device, non_blocking=True)

            if module.eval_mode == 'center_crop':
                pred, eval_mask = module.center_crop_eval(img, mask)
            elif module.eval_mode == 'sliding_window':
                pred = module.sliding_window_eval(img)
                eval_mask = mask
            else:
                pred = module(img, False).argmax(dim=1)
                eval_mask = mask

            valid = eval_mask != 255
            total_valid += int(valid.sum().item())
            if total_valid == 0:
                continue
            total_correct += int((pred[valid] == eval_mask[valid]).sum().item())

        if was_training:
            module.train()

        if total_valid == 0:
            return 0.0
        return float(total_correct / total_valid)

    def finalize(self, module, trainer, logger=None):
        device = module.device if isinstance(module.device, torch.device) else torch.device('cpu')
        generated, selected = self.aggregate_distributed(device=device)
        is_global_zero = getattr(trainer, 'is_global_zero', True)

        if not is_global_zero:
            return {
                'generated_pseudo_labels_total': int(generated),
                'selected_pseudo_labels_total': int(selected),
                'pseudo_label_selection_rate': float(selected / generated) if generated > 0 else 0.0,
                'pseudo_label_accuracy_on_val': None,
            }

        selection_rate = float(selected / generated) if generated > 0 else 0.0

        val_loader = None
        val_dataloaders = getattr(trainer, 'val_dataloaders', None)
        if isinstance(val_dataloaders, (list, tuple)) and len(val_dataloaders) > 0:
            val_loader = val_dataloaders[0]
        elif val_dataloaders is not None:
            val_loader = val_dataloaders

        pseudo_accuracy = self.evaluate_pseudo_accuracy(module, val_loader)

        summary = {
            'generated_pseudo_labels_total': int(generated),
            'selected_pseudo_labels_total': int(selected),
            'pseudo_label_selection_rate': float(selection_rate),
            'pseudo_label_accuracy_on_val': pseudo_accuracy,
            'per_epoch_cumulative': list(self.epoch_history),
        }

        os.makedirs(self.save_path, exist_ok=True)
        summary_path = os.path.join(self.save_path, 'pseudo_label_metrics_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)

        if logger is not None:
            logger.info('===== Pseudo Label Training Summary =====')
            logger.info(f"Generated pseudo labels (total): {summary['generated_pseudo_labels_total']}")
            logger.info(f"Selected pseudo labels (total): {summary['selected_pseudo_labels_total']}")
            logger.info(f"Pseudo-label selection rate: {summary['pseudo_label_selection_rate']:.6f}")
            if summary['pseudo_label_accuracy_on_val'] is None:
                logger.info('Pseudo-label accuracy on val: unavailable (no val dataloader)')
            else:
                logger.info(f"Pseudo-label accuracy on val: {summary['pseudo_label_accuracy_on_val']:.6f}")
            logger.info(f'Pseudo-label metrics summary saved to: {summary_path}')

        return summary