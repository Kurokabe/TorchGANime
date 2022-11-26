from pytorch_lightning.cli import LightningCLI

# simple demo classes for your convenience
from torchganime.models.vqgan import VQGAN
from torchganime.data.image import ImageData
from torchinfo import summary


def cli_main():
    cli = LightningCLI(VQGAN, ImageData)

    # summary(cli.model, input_size=(2, 3, 128, 256))


if __name__ == "__main__":
    cli_main()
