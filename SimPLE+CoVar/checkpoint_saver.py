import torch
import numpy as np
import os
import tempfile

from pathlib import Path
import re
import warnings

from utils import find_checkpoint_path, find_all_files, load_torch_checkpoint
from utils.metrics import MetricMode, MetricMonitor

# for type hint
from typing import Dict, Any, Optional, Union, List, Tuple
from simple_estimator import SimPLEEstimator
from utils import Logger


class CheckpointSaver:
    def __init__(self,
                 estimator: SimPLEEstimator,
                 logger: Logger,
                 checkpoint_metric: str,
                 best_checkpoint_str: str,
                 best_checkpoint_pattern: str,
                 latest_checkpoint_str: str,
                 latest_checkpoint_pattern: str,
                 delayed_best_model_saving: bool = True):
        """

        Args:
            estimator: Estimator, used to get experiment related data
            logger: Logger, used for logging
            checkpoint_metric: save model when the best value of this key has changed.
            best_checkpoint_str: path str format to save best checkpoint file
            best_checkpoint_pattern: regex pattern used to find the best checkpoint file
            latest_checkpoint_str: path str format to save best checkpoint file
            latest_checkpoint_pattern: regex pattern used to find the latest checkpoint file
            delayed_best_model_saving: if True, save best model after calling save_latest_checkpoint()
        """
        self.absolute_best_path = "best_checkpoint.pth"

        # metrics to keep track of
        self.monitor = MetricMonitor()
        self.monitor.track(key="mean_acc",
                           best_value=-np.inf,
                           mode=MetricMode.MAX,
                           prefix="test")
        self.monitor.track(key="mean_acc",
                           best_value=-np.inf,
                           mode=MetricMode.MAX,
                           prefix="validation")

        self.checkpoint_metric = checkpoint_metric

        # checkpoint path patterns
        self.best_checkpoint_str = best_checkpoint_str
        self.best_checkpoint_pattern = re.compile(best_checkpoint_pattern)

        self.latest_checkpoint_str = latest_checkpoint_str
        self.latest_checkpoint_pattern = re.compile(latest_checkpoint_pattern)

        # save estimator and logger
        # this will recover best metrics and register log hooks
        self.estimator = estimator
        self.logger = logger

        # assign flags
        self.delayed_save_best_model = delayed_best_model_saving
        self.is_best_model = False
        self.current_checkpoint_metric_value: Optional[float] = None

    @property
    def estimator(self) -> SimPLEEstimator:
        return self._estimator

    @estimator.setter
    def estimator(self, estimator: SimPLEEstimator) -> None:
        self._estimator = estimator

        # recover best value
        checkpoint_path = self.estimator.exp_args.checkpoint_path
        if checkpoint_path is not None:
            print(f"Recovering best metrics from {checkpoint_path}...")
            self.recover_metrics(load_torch_checkpoint(checkpoint_path, map_location=self.device))

    @property
    def logger(self) -> Logger:
        return self._logger

    @logger.setter
    def logger(self, logger: Logger) -> None:
        self._logger = logger

        # register log hooks
        print("Registering log hooks...")
        self.logger.register_log_hook(self.update_best_metric, logger=self.logger)

    @property
    def checkpoint_metric(self) -> str:
        return self._checkpoint_metric

    @checkpoint_metric.setter
    def checkpoint_metric(self, checkpoint_metric: str) -> None:
        assert checkpoint_metric in self.monitor, f"{checkpoint_metric} is not in metric monitor"

        self._checkpoint_metric = checkpoint_metric

    @property
    def log_dir(self) -> str:
        return self.estimator.exp_args.log_dir

    @property
    def best_full_checkpoint_str(self) -> str:
        return str(Path(self.log_dir) / self.best_checkpoint_str)

    @property
    def latest_full_checkpoint_str(self) -> str:
        return str(Path(self.log_dir) / self.latest_checkpoint_str)

    @property
    def device(self) -> torch.device:
        return self.estimator.device

    @property
    def global_step(self) -> int:
        return self.estimator.global_step

    @property
    def num_latest_checkpoints_kept(self) -> Optional[int]:
        value = getattr(self.estimator.exp_args, "num_latest_checkpoints_kept", 1)
        if value is None:
            return 1
        return int(value)

    @property
    def num_best_checkpoints_kept(self) -> int:
        value = getattr(self.estimator.exp_args, "num_best_checkpoints_kept", 5)
        if value is None:
            return 5
        return int(value)

    @property
    def is_save_latest_checkpoint(self) -> bool:
        return self.num_latest_checkpoints_kept > 0

    @property
    def is_remove_old_checkpoint(self) -> bool:
        return self.num_latest_checkpoints_kept > 0

    @property
    def is_save_best_checkpoint(self) -> bool:
        return self.num_best_checkpoints_kept > 0

    def save_checkpoint(self,
                        checkpoint: Dict[str, Any],
                        checkpoint_path: Union[str, Path],
                        is_logger_save: bool = False) -> Path:
        checkpoint_path = str(checkpoint_path)
        # Save atomically: write to a temp file then rename. Wrap in try/except so a full disk
        # won't crash the training loop; instead, we log the error and skip saving.
        tmp_dir = os.path.dirname(checkpoint_path) or '.'
        try:
            fd, tmp_name = tempfile.mkstemp(prefix=".ckpt_tmp_", dir=tmp_dir)
            os.close(fd)
            try:
                torch.save(checkpoint, tmp_name)
                # atomic replace
                os.replace(tmp_name, checkpoint_path)
            finally:
                # ensure no leftover temp file
                if os.path.exists(tmp_name):
                    try:
                        os.remove(tmp_name)
                    except Exception:
                        pass

            print(f"Checkpoint saved to \"{checkpoint_path}\"", flush=True)

            if is_logger_save:
                try:
                    self.logger.save(checkpoint_path)
                except Exception as e:
                    print(f"Logger save failed: {e}", flush=True)

            return Path(checkpoint_path)
        except Exception as e:
            # Likely disk full or IO error. Log and continue without raising to avoid crashing training.
            try:
                print(f"Warning: failed to save checkpoint to {checkpoint_path}: {e}", flush=True)
                # also try a direct save without atomic step as a last resort
                try:
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Checkpoint saved (fallback) to \"{checkpoint_path}\"", flush=True)
                    return Path(checkpoint_path)
                except Exception:
                    pass
            except Exception:
                pass

            return None

    def save_best_checkpoint(self,
                             checkpoint: Optional[Dict[str, Any]] = None,
                             is_logger_save: bool = False,
                             checkpoint_metric_value: Optional[float] = None,
                             **kwargs) -> Optional[Path]:
        if not self.is_save_best_checkpoint:
            self.is_best_model = False
            return None

        if checkpoint is None:
            checkpoint = self.get_checkpoint()
        else:
            checkpoint = dict(checkpoint)

        metric_value = checkpoint_metric_value
        if metric_value is None:
            metric_value = self.current_checkpoint_metric_value
        if metric_value is not None:
            checkpoint["checkpoint_metric_value"] = float(metric_value)

        checkpoint_path = self.save_checkpoint(checkpoint_path=self.best_full_checkpoint_str.format(**kwargs),
                                               checkpoint=checkpoint,
                                               is_logger_save=is_logger_save)
        # reset flag
        self.is_best_model = False

        return checkpoint_path

    def save_latest_checkpoint(self,
                               checkpoint: Optional[Dict[str, Any]] = None,
                               is_logger_save: bool = False,
                               **kwargs) -> Optional[Path]:
        checkpoint_path: Optional[Path] = None
        metric_value = self.current_checkpoint_metric_value

        if checkpoint is None:
            checkpoint = self.get_checkpoint()
        else:
            checkpoint = dict(checkpoint)

        if metric_value is not None:
            checkpoint["checkpoint_metric_value"] = float(metric_value)

        if self.is_save_latest_checkpoint:
            # save new checkpoint
            checkpoint_path = self.save_checkpoint(checkpoint_path=self.latest_full_checkpoint_str.format(**kwargs),
                                                   checkpoint=checkpoint,
                                                   is_logger_save=is_logger_save)

            # cleanup old checkpoints
            self.cleanup_checkpoints()

        if self.delayed_save_best_model and self.should_save_best_checkpoint(metric_value):
            self.save_best_checkpoint(
                checkpoint=checkpoint,
                is_logger_save=is_logger_save,
                checkpoint_metric_value=metric_value,
                **kwargs,
            )
            self.cleanup_best_checkpoints()

        self.is_best_model = False
        self.current_checkpoint_metric_value = None

        return checkpoint_path

    def get_checkpoint(self) -> Dict[str, Any]:
        checkpoint = self.estimator.get_checkpoint()

        # add best metrics
        checkpoint.update({"monitor_state": self.monitor.state_dict()})

        return checkpoint

    def update_best_checkpoint(self) -> None:
        """
        Update the logged metrics for the best checkpoint

        Returns:

        """
        best_checkpoint_path = self.find_best_checkpoint_path()

        if best_checkpoint_path is None:
            warnings.warn("Cannot find best checkpoint")
            return

        best_checkpoint_path = str(best_checkpoint_path)
        best_checkpoint = load_torch_checkpoint(best_checkpoint_path, map_location=self.device)

        # update best metrics
        best_checkpoint.update({"monitor_state": self.monitor.state_dict()})

        self.save_checkpoint(checkpoint_path=str(Path(self.log_dir) / self.absolute_best_path),
                             checkpoint=best_checkpoint)

    def find_best_checkpoint_path(self, checkpoint_dir: Optional[str] = None, ignore_absolute_best: bool = True) \
            -> Optional[Path]:
        if checkpoint_dir is None:
            checkpoint_dir = self.log_dir

        abs_best_path = Path(checkpoint_dir) / self.absolute_best_path

        if not ignore_absolute_best and abs_best_path.is_file():
            # if not ignoring absolute best path and the path is a file, return the absolute best file path
            return abs_best_path

        ranked_checkpoints = self.find_best_checkpoint_entries(checkpoint_dir=checkpoint_dir)
        checkpoint_path = ranked_checkpoints[0][0] if ranked_checkpoints else None

        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint_path(checkpoint_dir=checkpoint_dir)

        return checkpoint_path

    def find_latest_checkpoint_path(self, checkpoint_dir: Optional[str] = None) -> Optional[Path]:
        if checkpoint_dir is None:
            checkpoint_dir = self.log_dir

        return find_checkpoint_path(checkpoint_dir, step_filter=self.latest_checkpoint_pattern)

    def update_best_metric(self, log_info: Dict[str, Any], logger: Logger) -> None:
        if self.checkpoint_metric in log_info:
            metric_value = self._to_float(log_info[self.checkpoint_metric])
            if metric_value is not None:
                self.current_checkpoint_metric_value = metric_value

        updated_dict = self.monitor.update_metrics(log_info)

        for updated_key, new_best_value in updated_dict.items():
            metric_dict = self.monitor[updated_key]

            translated_key = metric_dict["key"]

            # if new_best_value is better than current best value
            logger.log({translated_key: new_best_value}, step=self.global_step)

            if self.checkpoint_metric == updated_key:
                self.is_best_model = True

        if not self.delayed_save_best_model and self.should_save_best_checkpoint(self.current_checkpoint_metric_value):
            self.save_best_checkpoint(
                global_step=self.global_step,
                checkpoint_metric_value=self.current_checkpoint_metric_value,
            )
            self.cleanup_best_checkpoints()
            self.is_best_model = False

    def recover_checkpoint(self, checkpoint: Dict[str, Any], recover_optimizer: bool = True,
                           recover_train_progress: bool = True) -> None:
        self.recover_metrics(checkpoint=checkpoint)

        self.estimator.load_checkpoint(checkpoint=checkpoint,
                                       recover_optimizer=recover_optimizer,
                                       recover_train_progress=recover_train_progress)

    def recover_metrics(self, checkpoint: Dict[str, Any]) -> None:
        if "monitor_state" in checkpoint:
            monitor_state = checkpoint["monitor_state"]
        else:
            # for backward compatibility
            monitor_state = {
                "validation/mean_acc": checkpoint.get("best_val_acc", -np.inf),
                "test/mean_acc": checkpoint.get("best_test_acc", -np.inf),
            }

        self.monitor.load_state_dict(monitor_state)

    def cleanup_checkpoints(self) -> None:
        if not self.is_remove_old_checkpoint:
            # do nothing if the model do not save latest checkpoints or if all checkpoints are kept
            return

        checkpoints = self.find_step_sorted_checkpoints(
            search_pattern=self.latest_checkpoint_pattern,
            checkpoint_dir=self.log_dir,
        )
        for checkpoint_path in checkpoints[self.num_latest_checkpoints_kept:]:
            checkpoint_path.unlink(missing_ok=True)

    def cleanup_best_checkpoints(self) -> None:
        if not self.is_save_best_checkpoint:
            return

        ranked_checkpoints = self.find_best_checkpoint_entries(checkpoint_dir=self.log_dir)
        for checkpoint_path, _ in ranked_checkpoints[self.num_best_checkpoints_kept:]:
            checkpoint_path.unlink(missing_ok=True)

    def should_save_best_checkpoint(self, metric_value: Optional[float]) -> bool:
        if not self.is_save_best_checkpoint or metric_value is None:
            return False

        ranked_checkpoints = self.find_best_checkpoint_entries(checkpoint_dir=self.log_dir)
        if len(ranked_checkpoints) < self.num_best_checkpoints_kept:
            return True

        worst_score = ranked_checkpoints[-1][1]
        return metric_value > worst_score

    def find_best_checkpoint_entries(self, checkpoint_dir: Optional[Union[str, Path]] = None) -> List[Tuple[Path, float]]:
        if checkpoint_dir is None:
            checkpoint_dir = self.log_dir

        ranked_checkpoints: List[Tuple[Path, float, int]] = []
        for checkpoint_path in find_all_files(checkpoint_dir, self.best_checkpoint_pattern):
            metric_value = self.load_checkpoint_metric_value(checkpoint_path)
            if metric_value is None:
                continue
            ranked_checkpoints.append(
                (checkpoint_path, metric_value, self.extract_step_from_path(checkpoint_path, self.best_checkpoint_pattern))
            )

        ranked_checkpoints.sort(key=lambda item: (item[1], item[2]), reverse=True)
        return [(checkpoint_path, metric_value) for checkpoint_path, metric_value, _ in ranked_checkpoints]

    def find_step_sorted_checkpoints(
        self,
        search_pattern: re.Pattern,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> List[Path]:
        if checkpoint_dir is None:
            checkpoint_dir = self.log_dir

        checkpoints = find_all_files(checkpoint_dir, search_pattern)
        checkpoints.sort(
            key=lambda checkpoint_path: self.extract_step_from_path(checkpoint_path, search_pattern),
            reverse=True,
        )
        return checkpoints

    def load_checkpoint_metric_value(self, checkpoint_path: Union[str, Path]) -> Optional[float]:
        try:
            checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
        except Exception:
            return None

        metric_value = checkpoint.get("checkpoint_metric_value")
        metric_value = self._to_float(metric_value)
        if metric_value is not None:
            return metric_value

        monitor_state = checkpoint.get("monitor_state", {})
        return self._to_float(monitor_state.get(self.checkpoint_metric))

    @staticmethod
    def extract_step_from_path(checkpoint_path: Union[str, Path], pattern: re.Pattern) -> int:
        search_result = re.search(pattern, Path(checkpoint_path).name)
        if search_result is None:
            return -1
        return int(search_result.group(1))

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

        chec