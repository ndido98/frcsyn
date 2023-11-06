from __future__ import annotations

from typing import Any, Literal

import torch
import torchmetrics as tm

from utils import kfold_indices, distance


def _compute_tpr_fpr_accuracies(distances: torch.Tensor, thresholds: torch.Tensor, actual: torch.Tensor, *, chunk_size: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # distances is a (n_samples,) tensor
    # thresholds is a (n_thresholds,) tensor
    # actual is a (n_samples,) tensor
    # Returns a tuple of (tpr, fpr, accuracies), each of which is a (n_thresholds,) tensor
    if thresholds.ndim == 0:
        thresholds = thresholds.unsqueeze(0)
    n_thresholds = thresholds.shape[0]
    thresholds_idx = torch.arange(n_thresholds, device=distances.device)
    tp = torch.zeros(n_thresholds, dtype=torch.int32, device=distances.device)
    fp = torch.zeros(n_thresholds, dtype=torch.int32, device=distances.device)
    tn = torch.zeros(n_thresholds, dtype=torch.int32, device=distances.device)
    fn = torch.zeros(n_thresholds, dtype=torch.int32, device=distances.device)
    for chunk_idx in thresholds_idx.chunk(chunk_size):
        chunk = thresholds[chunk_idx]
        predictions = distances.unsqueeze(0) < chunk.unsqueeze(1)
        tp[chunk_idx] = (predictions & actual.unsqueeze(0)).sum(dim=1, dtype=torch.int32)
        fp[chunk_idx] = (predictions & ~actual.unsqueeze(0)).sum(dim=1, dtype=torch.int32)
        tn[chunk_idx] = (~predictions & ~actual.unsqueeze(0)).sum(dim=1, dtype=torch.int32)
        fn[chunk_idx] = (~predictions & actual.unsqueeze(0)).sum(dim=1, dtype=torch.int32)
    tpr = torch.nan_to_num(tp / (tp + fn), nan=0.0, posinf=0.0, neginf=0.0)
    fpr = torch.nan_to_num(fp / (fp + tn), nan=0.0, posinf=0.0, neginf=0.0)
    accuracies = torch.nan_to_num((tp + tn) / (tp + tn + fp + fn), nan=0.0, posinf=0.0, neginf=0.0)
    return tpr, fpr, accuracies


def _calculate_roc(distances: torch.Tensor, is_same: torch.Tensor, n_folds: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert distances.ndim == 1
    thresholds: torch.Tensor = torch.sort(distances).values
    thresholds = torch.cat([thresholds, torch.tensor([thresholds[-1] + 1], device=thresholds.device)])
    n_thresholds = len(thresholds)
    tprs = torch.zeros(n_thresholds, device=distances.device)
    fprs = torch.zeros(n_thresholds, device=distances.device)
    accuracies = torch.zeros(n_folds, device=distances.device)
    best_thresholds = torch.zeros(n_folds, device=distances.device)
    for fold_idx, (train_idx, test_idx) in enumerate(kfold_indices(distances, n_folds)):
        _, _, train_accuracies = _compute_tpr_fpr_accuracies(distances[train_idx], thresholds, is_same[train_idx])
        best_threshold_idx = torch.argmax(train_accuracies)
        best_thresholds[fold_idx] = thresholds[best_threshold_idx]
        fold_tprs, fold_fprs, _ = _compute_tpr_fpr_accuracies(distances[test_idx], thresholds, is_same[test_idx])
        tprs += fold_tprs
        fprs += fold_fprs
        _, _, best_threshold_acc = _compute_tpr_fpr_accuracies(distances[test_idx], thresholds[best_threshold_idx], is_same[test_idx])
        accuracies[fold_idx] = best_threshold_acc
    tpr = tprs / n_folds
    fpr = fprs / n_folds
    return tpr, fpr, accuracies, best_thresholds


def _dim_zero_cat(tensors: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    if isinstance(tensors, torch.Tensor):
        return tensors
    else:
        return torch.cat(tensors, dim=0)


class EmbeddingAccuracy(tm.Metric):
    is_differentiable = False
    higher_is_better = None
    full_state_update = False

    def __init__(self, n_folds: int, distance_fn: Literal["euclidean", "cosine"] = "euclidean") -> None:
        super().__init__()
        self.n_folds = n_folds
        self.distance_fn = distance_fn
        self.add_state("distances", default=[], dist_reduce_fx="cat")
        self.add_state("is_same", default=[], dist_reduce_fx="cat")

    def update(self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor, is_same: torch.Tensor) -> None:
        assert embeddings_1.shape == embeddings_2.shape
        assert embeddings_1.shape[0] == is_same.shape[0]
        distances = distance(embeddings_1, embeddings_2, distance_fn=self.distance_fn)
        self.distances.append(distances)
        self.is_same.append(is_same)

    def compute(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        all_distances = _dim_zero_cat(self.distances)
        all_is_same = _dim_zero_cat(self.is_same)
        _ , _, accuracies, best_thresholds = _calculate_roc(all_distances, all_is_same, self.n_folds)
        return accuracies.mean(), best_thresholds.mean(), all_distances.flatten()
