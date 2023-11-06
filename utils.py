from __future__ import annotations

from typing import Generator, Literal

import torch
import torch.nn.functional as F


def normalize(input: torch.Tensor, dim: int = 1) -> torch.Tensor:
    norm = torch.norm(input, p=2, dim=dim, keepdim=True)
    return input / norm


def kfold_indices(data: torch.Tensor, folds: int) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
    indices = torch.arange(data.shape[0], device=data.device)
    fold_sizes = torch.full((folds,), data.shape[0] // folds, device=data.device, dtype=torch.int32)
    fold_sizes[: data.shape[0] % folds] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        train_idx = torch.cat((indices[:start], indices[stop:]))
        test_idx = indices[start:stop]
        yield train_idx, test_idx
        current = stop


def distance(x1: torch.Tensor, x2: torch.Tensor, distance_fn: Literal["euclidean", "cosine", "dot"] = "euclidean") -> torch.Tensor:
    if distance_fn == "euclidean":
        return torch.clip(F.pairwise_distance(x1, x2, p=2), min=0)
    elif distance_fn == "cosine":
        return torch.clip(1 - F.cosine_similarity(x1, x2), min=0, max=2)
    else:
        raise ValueError(f"Unknown distance function: {distance_fn}")