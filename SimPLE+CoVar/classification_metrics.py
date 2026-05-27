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
    def _merge_selection_stats(target: Dict[str, float], source: Dict[str, float]) -> None:
        for key in SELECTION_ACCUMULATOR_KEYS:
            target[key] += float(source.get(key, 0.0))

    def update_selection_stats(self, selection_state: Dict[str, Any]) -> None:
        max_confidence = selection_state["max_confidence"].detach().to(dtype=torch.float64).reshape(-1)
        scaled_residual_variance = selection_state["scaled_residual_variance"].detach().to(dtype=torch.float64).reshape(-1)
        weight = selection_state["weight"].detach().to(dtype=torch.float64).reshape(-1)
        selected_mask = selection_state["selected_mask"].detach().to(dtype=torch.bool).reshape(-1)

        sample_count = float(max_confidence.numel())
        selected_count = float(selected_mask.sum().item())
        rejected_count = sample_count - selected_count

        stats = self.epoch_selection_stats
        stats["batch_count"] += 1.0
        stats["sample_count"] += sample_count
        stats["selected_count"] += selected_count
        stats["rejected_count"] += rejected_count
        stats["max_confidence_sum"] += self._tensor_sum(max_confidence)
        stats["max_confidence_sq_sum"] += self._tensor_sq_sum(max_confidence)
        stats["scaled_residual_variance_sum"] += self._tensor_sum(scaled_residual_variance)
        stats["scaled_residual_variance_sq_sum"] += self._tensor_sq_sum(scaled_residual_variance)
        stats["weight_sum"] += self._tensor_sum(weight)
        stats["weight_sq_sum"] += self._tensor_sq_sum(weight)

        selected_confidence = max_confidence[selected_mask]
        selected_residual_variance = scaled_residual_variance[selected_mask]
        rejected_confidence = max_confidence[~selected_mask]
        rejected_residual_variance = scaled_residual_variance[~selected_mask]

        stats["selected_max_confidence_sum"] += self._tensor_sum(selected_confidence)
        stats["selected_max_confidence_sq_sum"] += self._tensor_sq_sum(selected_confidence)
        stats["selected_scaled_residual_variance_sum"] += self._tensor_sum(selected_residual_variance)
        stats["selected_scaled_residual_variance_sq_sum"] += self._tensor_sq_sum(selected_residual_variance)
        stats["rejected_max_confidence_sum"] += self._tensor_sum(rejected_confidence)
        stats["rejected_max_confidence_sq_sum"] += self._tensor_sq_sum(rejected_confidence)
        stats["rejected_scaled_residual_variance_sum"] += self._tensor_sum(rejected_residual_variance)
        stats["rejected_scaled_residual_variance_sq_sum"] += self._tensor_sq_sum(rejected_residual_variance)

        selected_cluster_count = float(selection_state.get("selected_cluster_count", 0.0))
        if selected_cluster_count > 0:
            selected_cluster_mean = selection_state["selected_cluster_mean"].detach().to(dtype=torch.float64)
            selected_cluster_var = selection_state["selected_cluster_var"].detach().to(dtype=torch.float64)
            selected_cluster_score = self._as_float(selection_state.get("selected_cluster_score", 0.0))
            stats["selected_cluster_weight"] += selected_cluster_count
            stats["selected_cluster_conf_mean_sum"] += float(selected_cluster_mean[0].item()) * selected_cluster_count
            stats["selected_cluster_res_mean_sum"] += float(selected_cluster_mean[1].item()) * selected_cluster_count
            stats["selected_cluster_conf_var_sum"] += float(selected_cluster_var[0].item()) * selected_cluster_count
            stats["selected_cluster_res_var_sum"] += float(selected_cluster_var[1].item()) * selected_cluster_count
            stats["selected_cluster_score_sum"] += selected_cluster_score * selected_cluster_count

        rejected_cluster_count = float(selection_state.get("rejected_cluster_count", 0.0))
        if rejected_cluster_count > 0:
            rejected_cluster_mean = selection_state["rejected_cluster_mean"].detach().to(dtype=torch.float64)
            rejected_cluster_var = selection_state["rejected_cluster_var"].detach().to(dtype=torch.float64)
            rejected_cluster_score = self._as_float(selection_state.get("rejected_cluster_score", 0.0))
            stats["rejected_cluster_weight"] += rejected_cluster_count
            stats["rejected_cluster_conf_mean_sum"] += float(rejected_cluster_mean[0].item()) * rejected_cluster_count
            stats["rejected_cluster_res_mean_sum"] += float(rejected_cluster_mean[1].item()) * rejected_cluster_count
            stats["rejected_cluster_conf_var_sum"] += float(rejected_cluster_var[0].item()) * rejected_cluster_count
            stats["rejected_cluster_res_var_sum"] += float(rejected_cluster_var[1].item()) * rejected_cluster_count
            stats["rejected_cluster_score_sum"] += rejected_cluster_score * rejected_cluster_count

    def update_batch(self, targets: torch.Tensor, true_targets: torch.Tensor, selected_mask: torch.Tensor) -> None:
        pseudo_labels = targets.argmax(dim=1)
        if true_targets.ndim > 1:
            true_labels = true_targets.argmax(dim=1)
        else:
            true_labels = true_targets

        selected_mask = selected_mask.to(dtype=torch.bool)
        correct_mask = pseudo_labels.eq(true_labels)
        selected_correct_mask = correct_mask & selected_mask

        total_samples = int(true_labels.numel())
        selected_samples = int(selected_mask.sum().item())
        correct_total_samples = int(correct_mask.sum().item())
        correct_selected_samples = int(selected_correct_mask.sum().item())

        self.epoch_total_samples += total_samples
        self.epoch_selected_samples += selected_samples
        self.epoch_correct_total_samples += correct_total_samples
        self.epoch_correct_selected_samples += correct_selected_samples

        total_per_class = torch.bincount(true_labels, minlength=self.num_classes)
        selected_per_class = torch.bincount(true_labels[selected_mask], minlength=self.num_classes)
        correct_selected_per_class = torch.bincount(true_labels[selected_correct_mask], minlength=self.num_classes)

        # Convert once to CPU to avoid many tiny host syncs from repeated .item() calls.
        total_per_class_list = [int(value) for value in total_per_class.detach().to(device="cpu").tolist()]
        selected_per_class_list = [int(value) for value in selected_per_class.detach().to(device="cpu").tolist()]
        correct_selected_per_class_list = [
            int(value) for value in correct_selected_per_class.detach().to(device="cpu").tolist()
        ]

        self.epoch_total_samples_per_class = [
            old + new for old, new in zip(self.epoch_total_samples_per_class, total_per_class_list)
        ]
        self.epoch_selected_samples_per_class = [
            old + new for old, new in zip(self.epoch_selected_samples_per_class, selected_per_class_list)
        ]
        self.epoch_correct_selected_samples_per_class = [
            old + new for old, new in zip(self.epoch_correct_selected_samples_per_class, correct_selected_per_class_list)
        ]

    def aggregate_scalars(self, values: Sequence[int], device: torch.device) -> List[int]:
        tensor = torch.tensor(list(values), dtype=torch.long, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return [int(value) for value in tensor.tolist()]

    def aggregate_vector(self, values: Sequence[int], device: torch.device) -> List[int]:
        tensor = torch.tensor(list(values), dtype=torch.long, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return [int(value) for value in tensor.tolist()]

    def aggregate_float_vector(self, values: Sequence[float], device: torch.device) -> List[float]:
        tensor = torch.tensor(list(values), dtype=torch.float64, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return [float(value) for value in tensor.tolist()]

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
        (
            epoch_total_samples,
            epoch_selected_samples,
            epoch_correct_total_samples,
            epoch_correct_selected_samples,
        ) = self.aggregate_scalars(
            [
                self.epoch_total_samples,
                self.epoch_selected_samples,
                self.epoch_correct_total_samples,
                self.epoch_correct_selected_samples,
            ],
            device=device,
        )
        epoch_total_samples_per_class = self.aggregate_vector(self.epoch_total_samples_per_class, device=device)
        epoch_selected_samples_per_class = self.aggregate_vector(self.epoch_selected_samples_per_class, device=device)
        epoch_correct_selected_samples_per_class = self.aggregate_vector(
            self.epoch_correct_selected_samples_per_class,
            device=device,
        )
        epoch_selection_stats_values = self.aggregate_float_vector(
            [self.epoch_selection_stats[key] for key in SELECTION_ACCUMULATOR_KEYS],
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
            "epoch_total_samples": self.epoch_total_samples,
            "epoch_selected_samples": self.epoch_selected_samples,
            "epoch_correct_total_samples": self.epoch_correct_total_samples,
            "epoch_correct_selected_samples": self.epoch_correct_selected_samples,
            "epoch_total_samples_per_class": list(self.epoch_total_samples_per_class),
            "epoch_selected_samples_per_class": list(self.epoch_selected_samples_per_class),
            "epoch_correct_selected_samples_per_class": list(self.epoch_correct_selected_samples_per_class),
            "total_selection_stats": dict(self.total_selection_stats),
            "epoch_selection_stats": dict(self.epoch_selection_stats),
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