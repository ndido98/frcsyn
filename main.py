import torch
from lightning.pytorch.cli import LightningCLI

from data import DataModule
from model import Model
from prediction_writer import PredictionWriter


def main() -> None:
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(model_class=Model, datamodule_class=DataModule, save_config_callback=False, parser_kwargs={"parser_mode": "omegaconf"})


if __name__ == "__main__":
    main()