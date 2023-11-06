from __future__ import annotations
from typing import Any, Literal

import torch
import torch.nn as nn
import lightning.pytorch as pl
import lightning.pytorch.loggers as pll
import wandb

from backbone import build_model
from head import AdaFace
from utils import distance
from metrics.embedding_roc import EmbeddingAccuracy


class Model(pl.LightningModule):
    def __init__(
        self,
        backbone: str,
        n_classes: int | None = None,
        margin: float = 0.4,
        h: float = 0.333,
        s: float = 64.0,
        t_alpha: float = 1.0,
        distance_fn: Literal["euclidean", "cosine"] = "euclidean",
        lr: float = 0.1,
        momentum: float = 0.9,
        lr_milestones: list[int] | None = None,
        lr_gamma: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.distance_fn = distance_fn
        self.lr = lr
        self.momentum = momentum
        self.lr_milestones = lr_milestones if lr_milestones is not None else [8, 12, 14]
        self.lr_gamma = lr_gamma

        self.backbone = build_model(backbone)
        if n_classes is not None:
            self.head = AdaFace(embedding_size=512, n_classes=n_classes, margin=margin, h=h, s=s, t_alpha=t_alpha)
        else:
            self.head = None

        self.loss = nn.CrossEntropyLoss()

        self.embedding_accuracy = EmbeddingAccuracy(n_folds=10, distance_fn=self.distance_fn)
        self.register_buffer("distance_threshold", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def on_train_start(self) -> None:
        if isinstance(self.logger, pll.WandbLogger) and self.trainer.global_rank == 0:
            self.logger.watch(self, log="all", log_graph=True)
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        embeddings, norms = self(x)
        cos_thetas = self.head(embeddings, norms, y)
        loss = self.loss(cos_thetas, y)
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x1, x2, issame = batch
        x1_embeddings, _ = self(x1)
        x2_embeddings, _ = self(x2)
        self.embedding_accuracy.update(x1_embeddings, x2_embeddings, issame)

    def on_validation_epoch_end(self) -> None:
        accuracy, best_threshold, all_distances = self.embedding_accuracy.compute()
        self.distance_threshold = best_threshold
        self.log("accuracy/val", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("best_threshold/val", best_threshold, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if isinstance(self.logger, pll.WandbLogger) and self.trainer.global_rank == 0:
            distances_list = [[d] for d in all_distances]
            table = wandb.Table(data=distances_list, columns=["distances"])
            self.logger.experiment.log(
                {
                    "distances/val": wandb.plot.histogram(
                        table,
                        "distances",
                        title="Distances",
                    )
                }
            )
        self.embedding_accuracy.reset()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = batch
        x1_embeddings, _ = self(x1)
        x2_embeddings, _ = self(x2)
        distances = distance(x1_embeddings, x2_embeddings, self.distance_fn)
        predictions = distances < self.distance_threshold
        scores = torch.sigmoid(self.distance_threshold - distances)
        return scores, predictions

    def configure_optimizers(self):
        paras_wo_bn, paras_only_bn = self.split_parameters(self.backbone)
        optimizer = torch.optim.SGD(
            [
                {
                    "params": paras_wo_bn + [self.head.kernel],
                    "weight_decay": 5e-4
                },
                {
                    "params": paras_only_bn,
                }
            ],
            lr=self.lr,
            momentum=self.momentum,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=self.lr_gamma)
        return [optimizer], [scheduler]

    def split_parameters(self, module: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        params_decay = []
        params_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                params_no_decay.extend([*m.parameters()])
            elif len(list(m.children())) == 0:
                params_decay.extend([*m.parameters()])
        assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
        return params_decay, params_no_decay