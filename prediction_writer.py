from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import lightning.pytorch as pl
import lightning.pytorch.callbacks as plc


class PredictionWriter(plc.BasePredictionWriter):
    def __init__(self, output_dir: Path, datasets_names: list[str], filename: str) -> None:
        super().__init__("batch_and_epoch")
        self.output_dir = output_dir
        self.datasets_names = datasets_names
        self.filename = filename
        self.outputs = {}

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: tuple[torch.Tensor, torch.Tensor],
        batch_indices: Sequence[int] | None,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if dataloader_idx not in self.outputs:
            self.outputs[dataloader_idx] = {}
        self.outputs[dataloader_idx][batch_idx] = prediction

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Sequence[Any] | None,
    ) -> None:
        for dataloader_idx, dataloader_outputs in self.outputs.items():
            dataloader_scores, dataloader_predictions = [], []
            for batch_idx in sorted(dataloader_outputs.keys()):
                batch_scores, batch_predictions = dataloader_outputs[batch_idx]
                dataloader_scores.append(batch_scores)
                dataloader_predictions.append(batch_predictions)
            dataloader_scores = torch.cat(dataloader_scores, dim=0).tolist()
            dataloader_predictions = torch.cat(dataloader_predictions, dim=0).tolist()
            output_file_path: Path = self.output_dir / self.datasets_names[dataloader_idx] / self.filename
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "w") as output_file:
                for score, prediction in zip(dataloader_scores, dataloader_predictions):
                    output_file.write(f"{score:.8f},{1 if prediction else 0}\n")