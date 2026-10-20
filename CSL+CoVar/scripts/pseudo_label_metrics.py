import json
import os
import numpy as np
import torch


def _default_monitor_thresholds():
    return [round(0.90 + 0.01 * i, 2) for i in range(10)]


def _normalize_thresholds(thresholds):
    if thresholds is None:
        return _default_monitor_thresholds()

    parsed = []
    if isinstance(thresholds, str):
        for token in thresholds.split(','):
            token = token.strip()
            if token:
                parsed.append(float(token))
    else:
        parsed = [float(x) for x in thresholds]

    valid = []
    for thr in parsed:
        if 0.0 <= thr <= 1.0:
            valid.append(round(thr, 4))

    if not valid:
        return _default_monitor_thresholds()
    return sorted(set(valid))


def _threshold_key(threshold):
    return f"{float(threshold):.2f}"


def _json_safe(value):
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


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
    def __init__(self, save_path, num_classes, monitor_thresholds=None, threshold_strategy='fixed'):
        self.save_path = save_path
        self.num_classes = num_classes
        self.threshold_strategy = threshold_strategy

        if threshold_strategy == 'dynamic':
            self.monitor_thresholds = []
            self.threshold_keys = ['dynamic']
        else:
            self.monitor_thresholds = _normalize_thresholds(monitor_thresholds)
            self.threshold_keys = [_threshold_key(thr) for thr in self.monitor_thresholds]

        self.epoch_history = []
        self._epoch_states = {k: self._empty_state() for k in self.threshold_keys}
        self._total_states = {k: self._empty_state() for k in self.threshold_keys}

    def _empty_state(self):
        return {
            'total_pixels': 0.0,
            'selected_pixels': 0.0,
            'full_correct_pixels': 0.0,
            'masked_correct_pixels': 0.0,
            'intersections': np.zeros(self.num_classes, dtype=np.float64),
            'unions': np.zeros(self.num_classes, dtype=np.float64),
            'selected_per_class': np.zeros(self.num_classes, dtype=np.float64),
            'correct_per_class': np.zeros(self.num_classes, dtype=np.float64),
            'gt_per_class': np.zeros(self.num_classes, dtype=np.float64),
            'covered_gt_per_class': np.zeros(self.num_classes, dtype=np.float64),
        }

    def _build_state(self, pseudo_labels, gt_labels, mask):
        valid = gt_labels != 255
        selected = mask & valid
        correct = pseudo_labels == gt_labels

        state = self._empty_state()
        state['total_pixels'] = float(valid.sum().item())
        state['selected_pixels'] = float(selected.sum().item())
        state['full_correct_pixels'] = float((correct & valid).sum().item())
        state['masked_correct_pixels'] = float((correct & selected).sum().item())

        for cls in range(self.num_classes):
            pred_cls = pseudo_labels == cls
            gt_cls = (gt_labels == cls) & valid
            pred_cls_selected = pred_cls & selected
            intersection = (pred_cls_selected & gt_cls).sum().item()
            union = (pred_cls_selected | gt_cls).sum().item()
            selected_cls = pred_cls_selected.sum().item()

            state['intersections'][cls] = float(intersection)
            state['unions'][cls] = float(union)
            state['selected_per_class'][cls] = float(selected_cls)
            state['correct_per_class'][cls] = float(intersection)
            state['gt_per_class'][cls] = float(gt_cls.sum().item())
            state['covered_gt_per_class'][cls] = float((gt_cls & selected).sum().item())

        return state

    def _accumulate_state(self, target, source):
        target['total_pixels'] += source['total_pixels']
        target['selected_pixels'] += source['selected_pixels']
        target['full_correct_pixels'] += source['full_correct_pixels']
        target['masked_correct_pixels'] += source['masked_correct_pixels']
        target['intersections'] += source['intersections']
        target['unions'] += source['unions']
        target['selected_per_class'] += source['selected_per_class']
        target['correct_per_class'] += source['correct_per_class']
        target['gt_per_class'] += source['gt_per_class']
        target['covered_gt_per_class'] += source['covered_gt_per_class']

    def _state_to_metrics(self, state):
        total_pixels = state['total_pixels']
        selected_pixels = state['selected_pixels']
        masked_correct_pixels = state['masked_correct_pixels']
        full_correct_pixels = state['full_correct_pixels']

        masked_acc = masked_correct_pixels / (selected_pixels + 1e-7)
        full_acc = full_correct_pixels / (total_pixels + 1e-7)
        coverage = selected_pixels / (total_pixels + 1e-7)

        valid_ious = state['unions'] > 0
        miou = float(np.mean(state['intersections'][valid_ious] / (state['unions'][valid_ious] + 1e-7))) if np.any(valid_ious) else 0.0

        cat_acc = []
        for cls in range(self.num_classes):
            cls_selected = state['selected_per_class'][cls]
            if cls_selected > 0:
                cat_acc.append(float(state['correct_per_class'][cls] / (cls_selected + 1e-7)))
            else:
                cat_acc.append(float('nan'))

        cat_coverage = []
        for cls in range(self.num_classes):
            gt_total = state['gt_per_class'][cls]
            if gt_total > 0:
                cat_coverage.append(float(state['covered_gt_per_class'][cls] / (gt_total + 1e-7)))
            else:
                cat_coverage.append(float('nan'))

        return {
            'masked_acc': float(masked_acc),
            'full_acc': float(full_acc),
            'coverage': float(coverage),
            'miou': float(miou),
            'cat_acc': cat_acc,
            'cat_coverage': cat_coverage,
        }

    def _reduce_state(self, state, device):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return state

        scalar_tensor = torch.tensor(
            [
                state['total_pixels'],
                state['selected_pixels'],
                state['full_correct_pixels'],
                state['masked_correct_pixels'],
            ],
            dtype=torch.float64,
            device=device,
        )
        intersections = torch.tensor(state['intersections'], dtype=torch.float64, device=device)
        unions = torch.tensor(state['unions'], dtype=torch.float64, device=device)
        selected_per_class = torch.tensor(state['selected_per_class'], dtype=torch.float64, device=device)
        correct_per_class = torch.tensor(state['correct_per_class'], dtype=torch.float64, device=device)
        gt_per_class = torch.tensor(state['gt_per_class'], dtype=torch.float64, device=device)
        covered_gt_per_class = torch.tensor(state['covered_gt_per_class'], dtype=torch.float64, device=device)

        torch.distributed.all_reduce(scalar_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(intersections, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(unions, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(selected_per_class, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(correct_per_class, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(gt_per_class, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(covered_gt_per_class, op=torch.distributed.ReduceOp.SUM)

        return {
            'total_pixels': float(scalar_tensor[0].item()),
            'selected_pixels': float(scalar_tensor[1].item()),
            'full_correct_pixels': float(scalar_tensor[2].item()),
            'masked_correct_pixels': float(scalar_tensor[3].item()),
            'intersections': intersections.cpu().numpy(),
            'unions': unions.cpu().numpy(),
            'selected_per_class': selected_per_class.cpu().numpy(),
            'correct_per_class': correct_per_class.cpu().numpy(),
            'gt_per_class': gt_per_class.cpu().numpy(),
            'covered_gt_per_class': covered_gt_per_class.cpu().numpy(),
        }

    def update_pseudo_label_metrics(self, pseudo_labels, gt_labels, max_probs, selection_mask=None):
        batch_metrics_by_threshold = {}

        if self.threshold_strategy == 'dynamic':
            if selection_mask is None:
                raise ValueError("selection_mask is required for dynamic threshold strategy")
            key = 'dynamic'
            state = self._build_state(pseudo_labels, gt_labels, selection_mask)
            metrics = self._state_to_metrics(state)
            self._accumulate_state(self._epoch_states[key], state)
            self._accumulate_state(self._total_states[key], state)
            batch_metrics_by_threshold[key] = metrics
        else:
            for threshold in self.monitor_thresholds:
                key = _threshold_key(threshold)
                threshold_mask = max_probs >= threshold
                state = self._build_state(pseudo_labels, gt_labels, threshold_mask)
                metrics = self._state_to_metrics(state)

                self._accumulate_state(self._epoch_states[key], state)
                self._accumulate_state(self._total_states[key], state)
                batch_metrics_by_threshold[key] = metrics

        return batch_metrics_by_threshold

    def _build_epoch_record(self, reduced_state, metrics):
        return {
            'total_pixels': int(reduced_state['total_pixels']),
            'selected_pixels': int(reduced_state['selected_pixels']),
            'masked_acc': metrics['masked_acc'],
            'full_acc': metrics['full_acc'],
            'coverage': metrics['coverage'],
            'miou': metrics['miou'],
            'cat_acc': metrics['cat_acc'],
            'cat_coverage': metrics['cat_coverage'],
        }

    def on_epoch_end(self, current_epoch, trainer, module, logger=None):
        device = module.device if isinstance(module.device, torch.device) else torch.device('cpu')
        epoch_metrics_by_threshold = {}

        if self.threshold_strategy == 'dynamic':
            key = 'dynamic'
            reduced_epoch_state = self._reduce_state(self._epoch_states[key], device)
            epoch_metrics = self._state_to_metrics(reduced_epoch_state)
            epoch_metrics_by_threshold[key] = epoch_metrics
            epoch_record = {
                'epoch': int(current_epoch),
                'metrics': self._build_epoch_record(reduced_epoch_state, epoch_metrics),
            }
        else:
            epoch_record = {'epoch': int(current_epoch), 'threshold_metrics': {}}
            for threshold in self.monitor_thresholds:
                key = _threshold_key(threshold)
                reduced_epoch_state = self._reduce_state(self._epoch_states[key], device)
                epoch_metrics = self._state_to_metrics(reduced_epoch_state)
                epoch_metrics_by_threshold[key] = epoch_metrics
                epoch_record['threshold_metrics'][key] = self._build_epoch_record(
                    reduced_epoch_state, epoch_metrics
                )

        is_global_zero = getattr(trainer, 'is_global_zero', True)
        if is_global_zero:
            self.epoch_history.append(epoch_record)

            if logger is not None:
                if self.threshold_strategy == 'dynamic':
                    metric = epoch_metrics_by_threshold['dynamic']
                    logger.info(
                        'Pseudo-label epoch summary | '
                        f"epoch={int(current_epoch)} | "
                        f"masked={metric['masked_acc']:.4f}, cov={metric['coverage']:.4f}, miou={metric['miou']:.4f}"
                    )
                else:
                    segments = []
                    for threshold in self.monitor_thresholds:
                        key = _threshold_key(threshold)
                        metric = epoch_metrics_by_threshold[key]
                        segments.append(
                            f"t={key}: masked={metric['masked_acc']:.4f}, cov={metric['coverage']:.4f}, miou={metric['miou']:.4f}"
                        )
                    logger.info(
                        'Pseudo-label epoch summary | '
                        f"epoch={int(current_epoch)} | "
                        + ' | '.join(segments)
                    )

        self._epoch_states = {k: self._empty_state() for k in self.threshold_keys}
        return epoch_metrics_by_threshold

    def finalize(self, module, trainer, logger=None):
        device = module.device if isinstance(module.device, torch.device) else torch.device('cpu')

        if self.threshold_strategy == 'dynamic':
            key = 'dynamic'
            reduced_total_state = self._reduce_state(self._total_states[key], device)
            summary_metrics = self._state_to_metrics(reduced_total_state)

            if not getattr(trainer, 'is_global_zero', True):
                return {key: summary_metrics}

            os.makedirs(self.save_path, exist_ok=True)
            summary_path = os.path.join(self.save_path, 'pseudo_label_metrics_summary.json')
            payload = {
                'threshold_strategy': 'dynamic',
                'summary_metrics': summary_metrics,
                'per_epoch_metrics': self.epoch_history,
            }
            with open(summary_path, 'w', encoding='utf-8') as handle:
                json.dump(_json_safe(payload), handle, indent=2)

            if logger is not None:
                logger.info('===== Pseudo Label Training Summary (dynamic) =====')
                logger.info(
                    f"masked={summary_metrics.get('masked_acc', 0):.6f}, "
                    f"full={summary_metrics.get('full_acc', 0):.6f}, "
                    f"coverage={summary_metrics.get('coverage', 0):.6f}, "
                    f"miou={summary_metrics.get('miou', 0):.6f}"
                )
                logger.info(f'Pseudo-label metrics summary saved to: {summary_path}')
            return {key: summary_metrics}

        # Fixed threshold mode
        summary_metrics_by_threshold = {}
        for threshold in self.monitor_thresholds:
            key = _threshold_key(threshold)
            reduced_total_state = self._reduce_state(self._total_states[key], device)
            summary_metrics_by_threshold[key] = self._state_to_metrics(reduced_total_state)

        if not getattr(trainer, 'is_global_zero', True):
            return summary_metrics_by_threshold

        os.makedirs(self.save_path, exist_ok=True)
        summary_path = os.path.join(self.save_path, 'pseudo_label_metrics_summary.json')
        payload = {
            'monitor_thresholds': self.monitor_thresholds,
            'summary_metrics_by_threshold': summary_metrics_by_threshold,
            'per_epoch_metrics': self.epoch_history,
        }
        with open(summary_path, 'w', encoding='utf-8') as handle:
            json.dump(_json_safe(payload), handle, indent=2)

        if logger is not None:
            logger.info('===== Pseudo Label Training Summary =====')
            for threshold in self.monitor_thresholds:
                key = _threshold_key(threshold)
                metric = summary_metrics_by_threshold[key]
                logger.info(
                    f"t={key} | masked={metric.get('masked_acc', 0):.6f}, "
                    f"full={metric.get('full_acc', 0):.6f}, "
                    f"coverage={metric.get('coverage', 0):.6f}, "
                    f"miou={metric.get('miou', 0):.6f}"
                )
            logger.info(f'Pseudo-label metrics summary saved to: {summary_path}')
        return summary_metrics_by_threshold
