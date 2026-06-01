import torch

# for type hint
from typing import Dict, Tuple
from torch import Tensor


_PAIR_INDICES_CACHE: Dict[Tuple[int, bool, str, int], Tensor] = {}


def _get_pair_cache_key(length: int, ordered_pair: bool, device: torch.device) -> Tuple[int, bool, str, int]:
    return (
        int(length),
        bool(ordered_pair),
        device.type,
        -1 if device.index is None else int(device.index),
    )


def _build_pair_indices(length: int, device: torch.device) -> Tensor:
    try:
        return torch.combinations(torch.arange(length, device=device), r=2)
    except RuntimeError:
        return torch.combinations(torch.arange(length), r=2).to(device=device)


def get_pair_indices(inputs: Tensor, ordered_pair: bool = False) -> Tensor:
    """
    Get pair indices between each element in input tensor

    Args:
        inputs: input tensor
        ordered_pair: if True, will return ordered pairs. (e.g. both inputs[i,j] and inputs[j,i] are included)

    Returns: a tensor of shape (K, 2) where K = choose(len(inputs),2) if ordered_pair is False.
        Else K = 2 * choose(len(inputs),2). Each row corresponds to two indices in inputs.

    """
    device = inputs.device
    length = len(inputs)

    if length < 2:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    cache_key = _get_pair_cache_key(length, ordered_pair, device)
    if cache_key in _PAIR_INDICES_CACHE:
        return _PAIR_INDICES_CACHE[cache_key]

    base_key = _get_pair_cache_key(length, False, device)
    indices = _PAIR_INDICES_CACHE.get(base_key)
    if indices is None:
        indices = _build_pair_indices(length, device)
        _PAIR_INDICES_CACHE[base_key] = indices

    if ordered_pair:
        # make pairs ordered (e.g. both (0,1) and (1,0) are included)
        indices = torch.cat((indices, indices[:, [1, 0]]), dim=0)
        _PAIR_INDICES_CACHE[cache_key] = indices

    return indices
