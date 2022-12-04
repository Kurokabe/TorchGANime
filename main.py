from pytorch_lightning.cli import LightningCLI

# simple demo classes for your convenience
from torchganime.models.vqgan import VQGAN
from torchganime.data.image import ImageData
from pytorch_lightning.callbacks import ModelCheckpoint

# from torchinfo import summary


def cli_main():
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val/rec_loss_epoch")
    cli = LightningCLI(
        VQGAN, ImageData, trainer_defaults={"callbacks": [checkpoint_callback]}
    )

    # summary(cli.model, input_size=(2, 3, 128, 256))


if __name__ == "__main__":
    cli_main()
