import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import math

import torch


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


SELECTION_ACCUMULATOR_KEYS = [
    "batch_count",
    "sample_count",
    "selected_count",
    "rejected_count",
    "max_confidence_sum",
    "max_confidence_sq_sum",
    "scaled_residual_variance_sum",
    "scaled_residual_variance_sq_sum",
    "weight_sum",
    "weight_sq_sum",
    "selected_max_confidence_sum",
    "selected_max_confidence_sq_sum",
    "selected_scaled_residual_variance_sum",
    "selected_scaled_residual_variance_sq_sum",
    "rejected_max_confidence_sum",
    "rejected_max_confidence_sq_sum",
    "rejected_scaled_residual_variance_sum",
    "rejected_scaled_residual_variance_sq_sum",
    "selected_cluster_weight",
    "selected_cluster_conf_mean_sum",
    "selected_cluster_res_mean_sum",
    "selected_cluster_conf_var_sum",
    "selected_cluster_res_var_sum",
    "selected_cluster_score_sum",
    "rejected_cluster_weight",
    "rejected_cluster_conf_mean_sum",
    "rejected_cluster_res_mean_sum",
    "rejected_cluster_conf_var_sum",
    "rejected_cluster_res_var_sum",
    "rejected_cluster_score_sum",
]

SELECTION_ACCUMULATOR_INDEX = {
    key: index for index, key in enumerate(SELECTION_ACCUMULATOR_KEYS)
}


def _build_empty_selection_accumulator() -> Dict[str, float]:
    return {key: 0.0 for key in SELECTION_ACCUMULATOR_KEYS}


def _safe_std(sum_value: float, sq_sum_value: float, count: float) -> float:
    if count <= 0:
        return 0.0
    mean_value = sum_value / count
    variance = max((sq_sum_value / count) - (mean_value ** 2), 0.0)
    return math.sqrt(variance)


class ClassificationPseudoLabelMetricsTracker:
    def __init__(
        self,
        save_path: Optional[str],
        num_classes: int,
        dataset_name: Optional[str] = None,
        use_csl: bool = True,
        threshold_strategy: str = "fixed",
        confidence_threshold: Optional[float] = None,
        select_mode: Optional[str] = None,
        select_lam: Optional[float] = None,
    ):
        self.save_path = save_path
        self.num_classes = int(num_classes)
        self.dataset_name = dataset_name
        self.use_csl = bool(use_csl)
        self.confidence_threshold = confidence_threshold
        self.threshold_strategy = str(threshold_strategy)
        self.select_mode = select_mode
        self.select_lam = select_lam
        self.class_names = [f"class_{index}" for index in range(self.num_classes)]

        self.total_samples = 0
        self.selected_samples = 0
        self.correct_total_samples = 0
        self.correct_selected_samples = 0
        self.total_samples_per_class = [0] * self.num_classes
        self.selected_samples_per_class = [0] * self.num_classes
        self.correct_selected_samples_per_class = [0] * self.num_classes

        self.epoch_total_samples = 0
        self.epoch_selected_samples = 0
        self.epoch_correct_total_samples = 0
        self.epoch_correct_selected_samples = 0
        self.epoch_total_samples_per_class = [0] * self.num_classes
        self.epoch_selected_samples_per_class = [0] * self.num_classes
        self.epoch_correct_selected_samples_per_class = [0] * self.num_classes

        self.per_epoch_metrics: List[Dict[str, Any]] = []
        self.total_selection_stats = _build_empty_selection_accumulator()
        self.epoch_selection_stats = _build_empty_selection_accumulator()

        self._epoch_scalar_counts_tensor: Optional[torch.Tensor] = None
        self._epoch_total_samples_per_class_tensor: Optional[torch.Tensor] = None
        self._epoch_selected_samples_per_class_tensor: Optional[torch.Tensor] = None
        self._epoch_correct_selected_samples_per_class_tensor: Optional[torch.Tensor] = None
        self._epoch_selection_stats_tensor: Optional[torch.Tensor] = None

    @property
    def summary_path(self) -> Optional[Path]:
        if not self.save_path:
            return None
        return Path(self.save_path) / "pseudo_label_metrics_summary.json"

    def set_save_path(self, save_path: Optional[str]) -> None:
        self.save_path = save_path

    @staticmethod
    def is_main_process() -> bool:
        return not (torch.distributed.is_available() and torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0

    def reset_epoch(self) -> None:
        self.epoch_total_samples = 0
        self.epoch_selected_samples = 0
        self.epoch_correct_total_samples = 0
        self.epoch_correct_selected_samples = 0
        self.epoch_total_samples_per_class = [0] * self.num_classes
        self.epoch_selected_samples_per_class = [0] * self.num_classes
        self.epoch_correct_selected_samples_per_class = [0] * self.num_classes
        self.epoch_selection_stats = _build_empty_selection_accumulator()

        if self._epoch_scalar_counts_tensor is not None:
            self._epoch_scalar_counts_tensor.zero_()
        if self._epoch_total_samples_per_class_tensor is not None:
            self._epoch_total_samples_per_class_tensor.zero_()
        if self._epoch_selected_samples_per_class_tensor is not None:
            self._epoch_selected_samples_per_class_tensor.zero_()
        if self._epoch_correct_selected_samples_per_class_tensor is not None:
            self._epoch_correct_selected_samples_per_class_tensor.zero_()
        if self._epoch_selection_stats_tensor is not None:
            self._epoch_selection_stats_tensor.zero_()

    def _ensure_epoch_tensors(self, device: torch.device) -> None:
        device = torch.device(device)

        if self._epoch_scalar_counts_tensor is None:
            self._epoch_scalar_counts_tensor = torch.tensor(
                [
                    self.epoch_total_samples,
                    self.epoch_selected_samples,
                    self.epoch_correct_total_samples,
                    self.epoch_correct_selected_samples,
                ],
                dtype=torch.long,
                device=device,
            )
            self._epoch_total_samples_per_class_tensor = torch.tensor(
                self.epoch_total_samples_per_class,
                dtype=torch.long,
                device=device,
            )
            self._epoch_selected_samples_per_class_tensor = torch.tensor(
                self.epoch_selected_samples_per_class,
                dtype=torch.long,
                device=device,
            )
            self._epoch_correct_selected_samples_per_class_tensor = torch.tensor(
                self.epoch_correct_selected_samples_per_class,
                dtype=torch.long,
                device=device,
            )
            self._epoch_selection_stats_tensor = torch.tensor(
                [self.epoch_selection_stats[key] for key in SELECTION_ACCUMULATOR_KEYS],
                dtype=torch.float64,
                device=device,
            )
            return

        if self._epoch_scalar_counts_tensor.device != device:
            self._epoch_scalar_counts_tensor = self._epoch_scalar_counts_tensor.to(device=device)
            self._epoch_total_samples_per_class_tensor = self._epoch_total_samples_per_class_tensor.to(device=device)
            self._epoch_selected_samples_per_class_tensor = self._epoch_selected_samples_per_class_tensor.to(device=device)
            self._epoch_correct_selected_samples_per_class_tensor = (
                self._epoch_correct_selected_samples_per_class_tensor.to(device=device)
            )
            self._epoch_selection_stats_tensor = self._epoch_selection_stats_tensor.to(device=device)

    def _get_epoch_scalar_state(self) -> List[int]:
        if self._epoch_scalar_counts_tensor is None:
            return [
                int(self.epoch_total_samples),
                int(self.epoch_selected_samples),
                int(self.epoch_correct_total_samples),
                int(self.epoch_correct_selected_samples),
            ]

        return [
            int(value) for value in self._epoch_scalar_counts_tensor.detach().to(device="cpu").tolist()
        ]

    def _get_epoch_total_samples_per_class_state(self) -> List[int]:
        if self._epoch_total_samples_per_class_tensor is None:
            return [int(value) for value in self.epoch_total_samples_per_class]

        return [
            int(value) for value in self._epoch_total_samples_per_class_tensor.detach().to(device="cpu").tolist()
        ]

    def _get_epoch_selected_samples_per_class_state(self) -> List[int]:
        if self._epoch_selected_samples_per_class_tensor is None:
            return [int(value) for value in self.epoch_selected_samples_per_class]

        return [
            int(value)
            for value in self._epoch_selected_samples_per_class_tensor.detach().to(device="cpu").tolist()
        ]

    def _get_epoch_correct_selected_samples_per_class_state(self) -> List[int]:
        if self._epoch_correct_selected_samples_per_class_tensor is None:
            return [int(value) for value in self.epoch_correct_selected_samples_per_class]

        return [
            int(value)
            for value in self._epoch_correct_selected_samples_per_class_tensor.detach().to(device="cpu").tolist()
        ]

    def _get_epoch_selection_stats_state(self) -> Dict[str, float]:
        if self._epoch_selection_stats_tensor is None:
            return dict(self.epoch_selection_stats)

        values = self._epoch_selection_stats_tensor.detach().to(device="cpu").tolist()
        return {
            key: float(value)
            for key, value in zip(SELECTION_ACCUMULATOR_KEYS, values)
        }

    @staticmethod
    def _tensor_sum(values: torch.Tensor) -> float:
        if values.numel() == 0:
            return 0.0
        values = values.detach().to(dtype=torch.float64).reshape(-1)
        return float(values.sum().item())

    @staticmethod
    def _tensor_sq_sum(values: torch.Tensor) -> float:
        if values.numel() == 0:
            return 0.0
        values = values.detach().to(dtype=torch.float64).reshape(-1)
        return float((values * values).sum().item())

    @staticmethod
    def _as_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        return float(value)

    @staticmethod
    def _as_tensor(value: Any, device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=device, dtype=dtype).reshape(())
        return torch.tensor(value, device=device, dtype=dtype)

    @staticmethod
    def _merge_selection_stats(target: Dict[str, float], source: Dict[str, float]) -> None:
        for key in SELECTION_ACCUMULATOR_KEYS:
            target[key] += float(source.get(key, 0.0))

    def update_selection_stats(self, selection_state: Dict[str, Any]) -> None:
        max_confidence = selection_state["max_confidence"].detach().to(dtype=torch.float64).reshape(-1)
        scaled_residual_variance = selection_state["scaled_residual_variance"].detach().to(dtype=torch.float64).reshape(-1)
        weight = selection_state["weight"].detach().to(dtype=torch.float64).reshape(-1)
        selected_mask = selection_state["selected_mask"].detach().to(dtype=torch.bool).reshape(-1)

        self._ensure_epoch_tensors(max_confidence.device)
        stats = self._epoch_selection_stats_tensor
        accumulator_index = SELECTION_ACCUMULATOR_INDEX

        sample_count = torch.tensor(float(max_confidence.numel()), device=max_confidence.device, dtype=torch.float64)
        selected_count = selected_mask.to(dtype=torch.float64).sum()
        rejected_count = sample_count - selected_count

        stats[accumulator_index["batch_count"]] += 1.0
        stats[accumulator_index["sample_count"]] += sample_count
        stats[accumulator_index["selected_count"]] += selected_count
        stats[accumulator_index["rejected_count"]] += rejected_count
        stats[accumulator_index["max_confidence_sum"]] += max_confidence.sum()
        stats[accumulator_index["max_confidence_sq_sum"]] += (max_confidence * max_confidence).sum()
        stats[accumulator_index["scaled_residual_variance_sum"]] += scaled_residual_variance.sum()
        stats[accumulator_index["scaled_residual_variance_sq_sum"]] += (
            scaled_residual_variance * scaled_residual_variance
        ).sum()
        stats[accumulator_index["weight_sum"]] += weight.sum()
        stats[accumulator_index["weight_sq_sum"]] += (weight * weight).sum()

        selected_confidence = max_confidence[selected_mask]
        selected_residual_variance = scaled_residual_variance[selected_mask]
        rejected_confidence = max_confidence[~selected_mask]
        rejected_residual_variance = scaled_residual_variance[~selected_mask]

        stats[accumulator_index["selected_max_confidence_sum"]] += selected_confidence.sum()
        stats[accumulator_index["selected_max_confidence_sq_sum"]] += (selected_confidence * selected_confidence).sum()
        stats[accumulator_index["selected_scaled_residual_variance_sum"]] += selected_residual_variance.sum()
        stats[accumulator_index["selected_scaled_residual_variance_sq_sum"]] += (
            selected_residual_variance * selected_residual_variance
        ).sum()
        stats[accumulator_index["rejected_max_confidence_sum"]] += rejected_confidence.sum()
        stats[accumulator_index["rejected_max_confidence_sq_sum"]] += (rejected_confidence * rejected_confidence).sum()
        stats[accumulator_index["rejected_scaled_residual_variance_sum"]] += rejected_residual_variance.sum()
        stats[accumulator_index["rejected_scaled_residual_variance_sq_sum"]] += (
            rejected_residual_variance * rejected_residual_variance
        ).sum()

        if "selected_cluster_mean" in selection_state and "selected_cluster_var" in selection_state:
            selected_cluster_count = self._as_tensor(
                selection_state.get("selected_cluster_count", 0.0),
                device=max_confidence.device,
            )
            selected_cluster_mean = selection_state["selected_cluster_mean"].detach().to(
                device=max_confidence.device,
                dtype=torch.float64,
            )
            selected_cluster_var = selection_state["selected_cluster_var"].detach().to(
                device=max_confidence.device,
                dtype=torch.float64,
            )
            selected_cluster_score = self._as_tensor(
                selection_state.get("selected_cluster_score", 0.0),
                device=max_confidence.device,
            )
            stats[accumulator_index["selected_cluster_weight"]] += selected_cluster_count
            stats[accumulator_index["selected_cluster_conf_mean_sum"]] += (
                selected_cluster_mean[0] * selected_cluster_count
            )
            stats[accumulator_index["selected_cluster_res_mean_sum"]] += (
                selected_cluster_mean[1] * selected_cluster_count
            )
            stats[accumulator_index["selected_cluster_conf_var_sum"]] += (
                selected_cluster_var[0] * selected_cluster_count
            )
            stats[accumulator_index["selected_cluster_res_var_sum"]] += (
                selected_cluster_var[1] * selected_cluster_count
            )
            stats[accumulator_index["selected_cluster_score_sum"]] += (
                selected_cluster_score * selected_cluster_count
            )

        if "rejected_cluster_mean" in selection_state and "rejected_cluster_var" in selection_state:
            rejected_cluster_count = self._as_tensor(
                selection_state.get("rejected_cluster_count", 0.0),
                device=max_confidence.device,
            )
            rejected_cluster_mean = selection_state["rejected_cluster_mean"].detach().to(
                device=max_confidence.device,
                dtype=torch.float64,
            )
            rejected_cluster_var = selection_state["rejected_cluster_var"].detach().to(
                device=max_confidence.device,
                dtype=torch.float64,
            )
            rejected_cluster_score = self._as_tensor(
                selection_state.get("rejected_cluster_score", 0.0),
                device=max_confidence.device,
            )
            stats[accumulator_index["rejected_cluster_weight"]] += rejected_cluster_count
            stats[accumulator_index["rejected_cluster_conf_mean_sum"]] += (
                rejected_cluster_mean[0] * rejected_cluster_count
            )
            stats[accumulator_index["rejected_cluster_res_mean_sum"]] += (
                rejected_cluster_mean[1] * rejected_cluster_count
            )
            stats[accumulator_index["rejected_cluster_conf_var_sum"]] += (
                rejected_cluster_var[0] * rejected_cluster_count
            )
            stats[accumulator_index["rejected_cluster_res_var_sum"]] += (
                rejected_cluster_var[1] * rejected_cluster_count
            )
            stats[accumulator_index["rejected_cluster_score_sum"]] += (
                rejected_cluster_score * rejected_cluster_count
            )

    def update_batch(self, targets: torch.Tensor, true_targets: torch.Tensor, selected_mask: torch.Tensor) -> None:
        pseudo_labels = targets.argmax(dim=1)
        if true_targets.ndim > 1:
            true_labels = true_targets.argmax(dim=1)
        else:
            true_labels = true_targets

        selected_mask = selected_mask.to(dtype=torch.bool)
        correct_mask = pseudo_labels.eq(true_labels)
        selected_correct_mask = correct_mask & selected_mask

        self._ensure_epoch_tensors(true_labels.device)
        self._epoch_scalar_counts_tensor[0] += int(true_labels.numel())
        self._epoch_scalar_counts_tensor[1] += selected_mask.sum(dtype=torch.long)
        self._epoch_scalar_counts_tensor[2] += correct_mask.sum(dtype=torch.long)
        self._epoch_scalar_counts_tensor[3] += selected_correct_mask.sum(dtype=torch.long)

        total_per_class = torch.bincount(true_labels, minlength=self.num_classes)
        selected_per_class = torch.bincount(true_labels[selected_mask], minlength=self.num_classes)
        correct_selected_per_class = torch.bincount(true_labels[selected_correct_mask], minlength=self.num_classes)

        self._epoch_total_samples_per_class_tensor.add_(total_per_class.to(dtype=torch.long))
        self._epoch_selected_samples_per_class_tensor.add_(selected_per_class.to(dtype=torch.long))
        self._epoch_correct_selected_samples_per_class_tensor.add_(correct_selected_per_class.to(dtype=torch.long))

    def aggregate_scalars(self, values: Sequence[int], device: torch.device) -> List[int]:
        if isinstance(values, torch.Tensor):
            tensor = values.detach().clone().to(device=device, dtype=torch.long)
        else:
            tensor = torch.tensor(list(values), dtype=torch.long, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return [int(value) for value in tensor.to(device="cpu").tolist()]

    def aggregate_vector(self, values: Sequence[int], device: torch.device) -> List[int]:
        if isinstance(values, torch.Tensor):
            tensor = values.detach().clone().to(device=device, dtype=torch.long)
        else:
            tensor = torch.tensor(list(values), dtype=torch.long, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return [int(value) for value in tensor.to(device="cpu").tolist()]

    def aggregate_float_vector(self, values: Sequence[float], device: torch.device) -> List[float]:
        if isinstance(values, torch.Tensor):
            tensor = values.detach().clone().to(device=device, dtype=torch.float64)
        else:
            tensor = torch.tensor(list(values), dtype=torch.float64, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return [float(value) for value in tensor.to(device="cpu").tolist()]

    def build_metrics(
        self,
        total_samples: int,
        selected_samples: int,
        correct_total_samples: int,
        correct_selected_samples: int,
        total_samples_per_class: Sequence[int],
        selected_samples_per_class: Sequence[int],
        correct_selected_samples_per_class: Sequence[int],
    ) -> Dict[str, Any]:
        cat_acc = [
            _safe_ratio(correct_selected_samples_per_class[index], selected_samples_per_class[index])
            for index in range(self.num_classes)
        ]
        cat_coverage = [
            _safe_ratio(selected_samples_per_class[index], total_samples_per_class[index])
            for index in range(self.num_classes)
        ]

        return {
            "total_samples": int(total_samples),
            "selected_samples": int(selected_samples),
            "correct_total_samples": int(correct_total_samples),
            "correct_selected_samples": int(correct_selected_samples),
            "full_acc": _safe_ratio(correct_total_samples, total_samples),
            "masked_acc": _safe_ratio(correct_selected_samples, selected_samples),
            "coverage": _safe_ratio(selected_samples, total_samples),
            "cat_acc": cat_acc,
            "cat_coverage": cat_coverage,
            "total_samples_per_class": [int(value) for value in total_samples_per_class],
            "selected_samples_per_class": [int(value) for value in selected_samples_per_class],
            "correct_selected_samples_per_class": [int(value) for value in correct_selected_samples_per_class],
        }

    def build_selection_stats(self, accumulator: Dict[str, float]) -> Optional[Dict[str, Any]]:
        sample_count = float(accumulator.get("sample_count", 0.0))
        if sample_count <= 0:
            return None

        selected_count = int(round(accumulator.get("selected_count", 0.0)))
        rejected_count = int(round(accumulator.get("rejected_count", 0.0)))
        batch_count = int(round(accumulator.get("batch_count", 0.0)))

        stats: Dict[str, Any] = {
            "threshold_strategy": self.threshold_strategy,
            "confidence_threshold": self.confidence_threshold,
            "select_mode": self.select_mode,
            "select_lam": self.select_lam,
            "num_batches": batch_count,
            "num_samples": int(round(sample_count)),
            "selected_count": selected_count,
            "rejected_count": rejected_count,
            "selected_ratio": _safe_ratio(selected_count, int(round(sample_count))),
            "weight_mean": accumulator["weight_sum"] / sample_count,
            "weight_std": _safe_std(accumulator["weight_sum"], accumulator["weight_sq_sum"], sample_count),
            "max_confidence_mean": accumulator["max_confidence_sum"] / sample_count,
            "max_confidence_std": _safe_std(
                accumulator["max_confidence_sum"],
                accumulator["max_confidence_sq_sum"],
                sample_count,
            ),
            "scaled_residual_variance_mean": accumulator["scaled_residual_variance_sum"] / sample_count,
            "scaled_residual_variance_std": _safe_std(
                accumulator["scaled_residual_variance_sum"],
                accumulator["scaled_residual_variance_sq_sum"],
                sample_count,
            ),
        }

        if selected_count > 0:
            stats["selected_max_confidence_mean"] = accumulator["selected_max_confidence_sum"] / selected_count
            stats["selected_max_confidence_std"] = _safe_std(
                accumulator["selected_max_confidence_sum"],
                accumulator["selected_max_confidence_sq_sum"],
                selected_count,
            )
            stats["selected_scaled_residual_variance_mean"] = (
                accumulator["selected_scaled_residual_variance_sum"] / selected_count
            )
            stats["selected_scaled_residual_variance_std"] = _safe_std(
                accumulator["selected_scaled_residual_variance_sum"],
                accumulator["selected_scaled_residual_variance_sq_sum"],
                selected_count,
            )

        if rejected_count > 0:
            stats["rejected_max_confidence_mean"] = accumulator["rejected_max_confidence_sum"] / rejected_count
            stats["rejected_max_confidence_std"] = _safe_std(
                accumulator["rejected_max_confidence_sum"],
                accumulator["rejected_max_confidence_sq_sum"],
                rejected_count,
            )
            stats["rejected_scaled_residual_variance_mean"] = (
                accumulator["rejected_scaled_residual_variance_sum"] / rejected_count
            )
            stats["rejected_scaled_residual_variance_std"] = _safe_std(
                accumulator["rejected_scaled_residual_variance_sum"],
                accumulator["rejected_scaled_residual_variance_sq_sum"],
                rejected_count,
            )

        selected_cluster_weight = accumulator.get("selected_cluster_weight", 0.0)
        if selected_cluster_weight > 0:
            stats["selected_cluster_mean"] = {
                "confidence": accumulator["selected_cluster_conf_mean_sum"] / selected_cluster_weight,
                "scaled_residual_variance": accumulator["selected_cluster_res_mean_sum"] / selected_cluster_weight,
            }
            stats["selected_cluster_var"] = {
                "confidence": accumulator["selected_cluster_conf_var_sum"] / selected_cluster_weight,
                "scaled_residual_variance": accumulator["selected_cluster_res_var_sum"] / selected_cluster_weight,
            }
            stats["selected_cluster_score_mean"] = (
                accumulator["selected_cluster_score_sum"] / selected_cluster_weight
            )

        rejected_cluster_weight = accumulator.get("rejected_cluster_weight", 0.0)
        if rejected_cluster_weight > 0:
            stats["rejected_cluster_mean"] = {
                "confidence": accumulator["rejected_cluster_conf_mean_sum"] / rejected_cluster_weight,
                "scaled_residual_variance": accumulator["rejected_cluster_res_mean_sum"] / rejected_cluster_weight,
            }
            stats["rejected_cluster_var"] = {
                "confidence": accumulator["rejected_cluster_conf_var_sum"] / rejected_cluster_weight,
                "scaled_residual_variance": accumulator["rejected_cluster_res_var_sum"] / rejected_cluster_weight,
            }
            stats["rejected_cluster_score_mean"] = (
                accumulator["rejected_cluster_score_sum"] / rejected_cluster_weight
            )

        return stats

    def build_payload(self) -> Dict[str, Any]:
        summary_metrics = self.build_metrics(
            total_samples=self.total_samples,
            selected_samples=self.selected_samples,
            correct_total_samples=self.correct_total_samples,
            correct_selected_samples=self.correct_selected_samples,
            total_samples_per_class=self.total_samples_per_class,
            selected_samples_per_class=self.selected_samples_per_class,
            correct_selected_samples_per_class=self.correct_selected_samples_per_class,
        )

        payload = {
            "task": "classification",
            "dataset_name": self.dataset_name,
            "num_classes": self.num_classes,
            "use_csl": self.use_csl,
            "threshold_strategy": self.threshold_strategy,
            "confidence_threshold": self.confidence_threshold,
            "select_mode": self.select_mode,
            "select_lam": self.select_lam,
            "class_names": list(self.class_names),
            "summary_metrics": summary_metrics,
            "per_epoch_metrics": list(self.per_epoch_metrics),
        }

        summary_selection_stats = self.build_selection_stats(self.total_selection_stats)
        if summary_selection_stats is not None:
            payload["summary_selection_stats"] = summary_selection_stats

        return payload

    def write_summary(self) -> Optional[Path]:
        summary_path = self.summary_path
        if summary_path is None:
            return None

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(self.build_payload(), handle, indent=2)
        return summary_path

    def on_epoch_end(self, current_epoch: int, device: torch.device) -> Optional[Path]:
        self._ensure_epoch_tensors(device)
        (
            epoch_total_samples,
            epoch_selected_samples,
            epoch_correct_total_samples,
            epoch_correct_selected_samples,
        ) = self.aggregate_scalars(
            self._epoch_scalar_counts_tensor,
            device=device,
        )
        epoch_total_samples_per_class = self.aggregate_vector(self._epoch_total_samples_per_class_tensor, device=device)
        epoch_selected_samples_per_class = self.aggregate_vector(
            self._epoch_selected_samples_per_class_tensor,
            device=device,
        )
        epoch_correct_selected_samples_per_class = self.aggregate_vector(
            self._epoch_correct_selected_samples_per_class_tensor,
            device=device,
        )
        epoch_selection_stats_values = self.aggregate_float_vector(
            self._epoch_selection_stats_tensor,
            device=device,
        )
        epoch_selection_stats = {
            key: value for key, value in zip(SELECTION_ACCUMULATOR_KEYS, epoch_selection_stats_values)
        }

        summary_path = None
        if self.is_main_process():
            self.total_samples += epoch_total_samples
            self.selected_samples += epoch_selected_samples
            self.correct_total_samples += epoch_correct_total_samples
            self.correct_selected_samples += epoch_correct_selected_samples

            for class_index in range(self.num_classes):
                self.total_samples_per_class[class_index] += epoch_total_samples_per_class[class_index]
                self.selected_samples_per_class[class_index] += epoch_selected_samples_per_class[class_index]
                self.correct_selected_samples_per_class[class_index] += epoch_correct_selected_samples_per_class[class_index]

            epoch_metrics = self.build_metrics(
                total_samples=epoch_total_samples,
                selected_samples=epoch_selected_samples,
                correct_total_samples=epoch_correct_total_samples,
                correct_selected_samples=epoch_correct_selected_samples,
                total_samples_per_class=epoch_total_samples_per_class,
                selected_samples_per_class=epoch_selected_samples_per_class,
                correct_selected_samples_per_class=epoch_correct_selected_samples_per_class,
            )
            self._merge_selection_stats(self.total_selection_stats, epoch_selection_stats)

            record = {"epoch": int(current_epoch), "metrics": epoch_metrics}
            selection_stats = self.build_selection_stats(epoch_selection_stats)
            if selection_stats is not None:
                record["selection_stats"] = selection_stats
            if self.per_epoch_metrics and int(self.per_epoch_metrics[-1]["epoch"]) == int(current_epoch):
                self.per_epoch_metrics[-1] = record
            else:
                self.per_epoch_metrics.append(record)
            summary_path = self.write_summary()

        self.reset_epoch()
        return summary_path

    def state_dict(self) -> Dict[str, Any]:
        (
            epoch_total_samples,
            epoch_selected_samples,
            epoch_correct_total_samples,
            epoch_correct_selected_samples,
        ) = self._get_epoch_scalar_state()

        return {
            "save_path": self.save_path,
            "num_classes": self.num_classes,
            "dataset_name": self.dataset_name,
            "use_csl": self.use_csl,
            "confidence_threshold": self.confidence_threshold,
            "threshold_strategy": self.threshold_strategy,
            "select_mode": self.select_mode,
            "select_lam": self.select_lam,
            "class_names": list(self.class_names),
            "total_samples": self.total_samples,
            "selected_samples": self.selected_samples,
            "correct_total_samples": self.correct_total_samples,
            "correct_selected_samples": self.correct_selected_samples,
            "total_samples_per_class": list(self.total_samples_per_class),
            "selected_samples_per_class": list(self.selected_samples_per_class),
            "correct_selected_samples_per_class": list(self.correct_selected_samples_per_class),
            "epoch_total_samples": epoch_total_samples,
            "epoch_selected_samples": epoch_selected_samples,
            "epoch_correct_total_samples": epoch_correct_total_samples,
            "epoch_correct_selected_samples": epoch_correct_selected_samples,
            "epoch_total_samples_per_class": self._get_epoch_total_samples_per_class_state(),
            "epoch_selected_samples_per_class": self._get_epoch_selected_samples_per_class_state(),
            "epoch_correct_selected_samples_per_class": self._get_epoch_correct_selected_samples_per_class_state(),
            "total_selection_stats": dict(self.total_selection_stats),
            "epoch_selection_stats": self._get_epoch_selection_stats_state(),
            "per_epoch_metrics": list(self.per_epoch_metrics),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.save_path = state_dict.get("save_path", self.save_path)
        self.dataset_name = state_dict.get("dataset_name", self.dataset_name)
        self.use_csl = bool(state_dict.get("use_csl", self.use_csl))
        self.confidence_threshold = state_dict.get("confidence_threshold", self.confidence_threshold)
        self.threshold_strategy = state_dict.get("threshold_strategy", self.threshold_strategy)
        self.select_mode = state_dict.get("select_mode", self.select_mode)
        self.select_lam = state_dict.get("select_lam", self.select_lam)
        self.class_names = list(state_dict.get("class_names", self.class_names))

        self.total_samples = int(state_dict.get("total_samples", 0))
        self.selected_samples = int(state_dict.get("selected_samples", 0))
        self.correct_total_samples = int(state_dict.get("correct_total_samples", 0))
        self.correct_selected_samples = int(state_dict.get("correct_selected_samples", 0))
        self.total_samples_per_class = list(state_dict.get("total_samples_per_class", [0] * self.num_classes))
        self.selected_samples_per_class = list(state_dict.get("selected_samples_per_class", [0] * self.num_classes))
        self.correct_selected_samples_per_class = list(
            state_dict.get("correct_selected_samples_per_class", [0] * self.num_classes)
        )

        self.epoch_total_samples = int(state_dict.get("epoch_total_samples", 0))
        self.epoch_selected_samples = int(state_dict.get("epoch_selected_samples", 0))
        self.epoch_correct_total_samples = int(state_dict.get("epoch_correct_total_samples", 0))
        self.epoch_correct_selected_samples = int(state_dict.get("epoch_correct_selected_samples", 0))
        self.epoch_total_samples_per_class = list(state_dict.get("epoch_total_samples_per_class", [0] * self.num_classes))
        self.epoch_selected_samples_per_class = list(
            state_dict.get("epoch_selected_samples_per_class", [0] * self.num_classes)
        )
        self.epoch_correct_selected_samples_per_class = list(
            state_dict.get("epoch_correct_selected_samples_per_class", [0] * self.num_classes)
        )
        self.total_selection_stats = dict(state_dict.get("total_selection_stats", _build_empty_selection_accumulator()))
        self.epoch_selection_stats = dict(state_dict.get("epoch_selection_stats", _build_empty_selection_accumulator()))
        self.per_epoch_metrics = list(state_dict.get("per_epoch_metrics", []))

        self._epoch_scalar_counts_tensor = None
        self._epoch_total_samples_per_class_tensor = None
        self._epoch_selected_samples_per_class_tensor = None
        self._epoch_correct_selected_samples_per_class_tensor = None
        self._epoch_selection_stats_tensor = None

    def load_from_summary_file(self, summary_path: Optional[Path] = None) -> bool:
        if summary_path is None:
            summary_path = self.summary_path
        if summary_path is None or not summary_path.exists():
            return False

        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.dataset_name = payload.get("dataset_name", self.dataset_name)
        self.use_csl = bool(payload.get("use_csl", self.use_csl))
        self.confidence_threshold = payload.get("confidence_threshold", self.confidence_threshold)
        self.threshold_strategy = payload.get("threshold_strategy", self.threshold_strategy)
        self.select_mode = payload.get("select_mode", self.select_mode)
        self.select_lam = payload.get("select_lam", self.select_lam)
        self.class_names = list(payload.get("class_names", self.class_names))
        self.per_epoch_metrics = list(payload.get("per_epoch_metrics", []))

        summary_metrics = payload.get("summary_metrics", {})
        self.total_samples = int(summary_metrics.get("total_samples", 0))
        self.selected_samples = int(summary_metrics.get("selected_samples", 0))
        self.correct_total_samples = int(summary_metrics.get("correct_total_samples", 0))
        self.correct_selected_samples = int(summary_metrics.get("correct_selected_samples", 0))
        self.total_samples_per_class = list(summary_metrics.get("total_samples_per_class", [0] * self.num_classes))
        self.selected_samples_per_class = list(summary_metrics.get("selected_samples_per_class", [0] * self.num_classes))
        self.correct_selected_samples_per_class = list(
            summary_metrics.get("correct_selected_samples_per_class", [0] * self.num_classes)
        )

        self.total_selection_stats = _build_empty_selection_accumulator()
        self.reset_epoch()
        return True