from pytorch_lightning.cli import LightningCLI

from torchganime.models.ganime import GANime
from torchganime.data.dataloader.video import VideoData
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DeepSpeedStrategy
import torch


def cli_main():
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val/loss")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    torch.set_float32_matmul_precision("medium")

    cli = LightningCLI(
        GANime,
        VideoData,
        trainer_defaults={
            "callbacks": [checkpoint_callback, lr_monitor],
            "strategy": DeepSpeedStrategy(
                stage=2,
                offload_optimizer=True,
                offload_parameters=False,
            ),
        },
    )


if __name__ == "__main__":
    cli_main()
