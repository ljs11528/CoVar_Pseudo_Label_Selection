import json
import os
import numpy as np
import torch


def compute_metrics(pseudo_labels, gt_labels, mask, num_classes):
    """
    pseudo_labels: [B, H, W] 预测标签
    gt_labels: [B, H, W] 上帝视角标签
    mask: [B, H, W] 置信度阈值掩码 (True=选中)
    num_classes: 类别数
    """
    ignore = (gt_labels == 255)
    valid = ~ignore
    total_pixels = valid.sum().item()
    valid_pseudo_pixels = (mask & valid).sum().item()
    correct_pixels = ((pseudo_labels == gt_labels) & mask & valid).sum().item()
    full_correct_pixels = ((pseudo_labels == gt_labels) & valid).sum().item()

    # 精度
    masked_acc = correct_pixels / (valid_pseudo_pixels + 1e-7)
    full_acc = full_correct_pixels / (total_pixels + 1e-7)
    coverage = valid_pseudo_pixels / (total_pixels + 1e-7)

    # mIoU
    ious = []
    for cls in range(num_classes):
        if cls == 255:
            continue
        pred_mask = (pseudo_labels == cls) & mask & valid
        gt_mask = (gt_labels == cls) & valid
        intersection = (pred_mask & gt_mask).sum().item()
        union = (pred_mask | gt_mask).sum().item()
        if union > 0:
            ious.append(intersection / (union + 1e-7))
    miou = np.mean(ious) if ious else 0.0

    # 各类别精度
    cat_acc = []
    for cls in range(num_classes):
        if cls == 255:
            continue
        cls_mask = (pseudo_labels == cls) & mask & valid
        cls_gt = (gt_labels == cls) & mask & valid
        cls_total = cls_mask.sum().item()
        cls_correct = (cls_mask & cls_gt).sum().item()
        if cls_total > 0:
            cat_acc.append(cls_correct / (cls_total + 1e-7))
        else:
            cat_acc.append(float('nan'))

    return {
        'masked_acc': masked_acc,
        'full_acc': full_acc,
        'coverage': coverage,
        'miou': miou,
        'cat_acc': cat_acc,
    }


class PseudoLabelMetricsTracker:
    def __init__(self, save_path, num_classes):
        self.save_path = save_path
        self.num_classes = num_classes
        self.metrics_history = []

    def update_pseudo_label_metrics(self, pseudo_labels, gt_labels, mask):
        # 计算并保存本 batch 的伪标签指标
        metrics = compute_metrics(pseudo_labels, gt_labels, mask, self.num_classes)
        self.metrics_history.append(metrics)
        return metrics

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
        # 汇总所有 batch 的伪标签指标
        all_metrics = self.metrics_history
        mean_metrics = {}
        if all_metrics:
            for k in all_metrics[0].keys():
                vals = [m[k] for m in all_metrics if not isinstance(m[k], list)]
                mean_metrics[k] = float(np.nanmean(vals))
            # 类别精度单独处理
            cat_accs = [m['cat_acc'] for m in all_metrics]
            mean_metrics['cat_acc'] = np.nanmean(cat_accs, axis=0).tolist()

        os.makedirs(self.save_path, exist_ok=True)
        summary_path = os.path.join(self.save_path, 'pseudo_label_metrics_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as handle:
            json.dump({'mean_metrics': mean_metrics, 'all_metrics': all_metrics}, handle, indent=2)

        if logger is not None:
            logger.info('===== Pseudo Label Training Summary =====')
            logger.info(f"Masked Pseudo-Label Acc: {mean_metrics.get('masked_acc', 0):.6f}")
            logger.info(f"Full Pseudo-Label Acc: {mean_metrics.get('full_acc', 0):.6f}")
            logger.info(f"Coverage: {mean_metrics.get('coverage', 0):.6f}")
            logger.info(f"mIoU: {mean_metrics.get('miou', 0):.6f}")
            logger.info(f"Categorical Acc: {mean_metrics.get('cat_acc', [])}")
            logger.info(f'Pseudo-label metrics summary saved to: {summary_path}')
        return mean_metrics
