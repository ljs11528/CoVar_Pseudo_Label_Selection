import torch
import numpy as np
import yaml

from pathlib import Path
import re

# for type hint
from typing import Union, Pattern, Optional, Dict, Any, List


def load_torch_checkpoint(
    checkpoint_path: Union[str, Path],
    map_location: Optional[Union[str, torch.device, Dict[str, str], Dict[torch.device, torch.device]]] = None,
):
    load_kwargs = dict(map_location=map_location)

    try:
        return torch.load(checkpoint_path, weights_only=False, **load_kwargs)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        return torch.load(checkpoint_path, **load_kwargs)


def find_checkpoint_path(checkpoint_dir: Union[str, Path], step_filter: Union[Pattern, str]) -> Optional[Path]:
    checkpoint_dir_path = Path(checkpoint_dir)
    if not checkpoint_dir_path.is_dir():
        return None

    output_file = None
    max_step_num = -np.inf

    for file_item in checkpoint_dir_path.iterdir():
        if not file_item.is_file():
            continue

        search_result = re.search(step_filter, file_item.name)
        if search_result is None:
            continue

        step_num = int(search_result.group(1))
        if step_num > max_step_num:
            max_step_num = step_num
            output_file = file_item

    return output_file


def find_latest_directory(search_dir: Union[str, Path]) -> Optional[Path]:
    search_dir_path = Path(search_dir)
    if not search_dir_path.is_dir():
        return None

    candidate_dirs = [child for child in search_dir_path.iterdir() if child.is_dir()]
    if not candidate_dirs:
        return None

    return max(candidate_dirs, key=lambda path: (path.stat().st_mtime, path.name))


def find_latest_checkpoint_from_latest_log_dir(
    search_dir: Union[str, Path],
    step_filter: Union[Pattern, str],
) -> Optional[Path]:
    search_dir_path = Path(search_dir)
    if not search_dir_path.is_dir():
        return None

    direct_checkpoint_path = find_checkpoint_path(search_dir_path, step_filter)
    if direct_checkpoint_path is not None:
        return direct_checkpoint_path

    candidate_dirs = sorted(
        (child for child in search_dir_path.iterdir() if child.is_dir()),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )

    for run_dir in candidate_dirs:
        checkpoint_path = find_checkpoint_path(run_dir, step_filter)
        if checkpoint_path is not None:
            return checkpoint_path

    return None


def read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_all_files(checkpoint_dir: Union[str, Path], search_pattern: Union[Pattern, str]) -> List[Path]:
    checkpoint_dir_path = Path(checkpoint_dir)

    return [file_item for file_item in checkpoint_dir_path.iterdir()
            if file_item.is_file() and re.search(pattern=search_pattern, string=file_item.name) is not None]
