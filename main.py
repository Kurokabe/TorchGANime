from pytorch_lightning.cli import LightningCLI

# simple demo classes for your convenience
from torchganime.models.vqgan import VQGAN
from torchganime.data.image import ImageData


def cli_main():
    cli = LightningCLI(VQGAN, ImageData)


if __name__ == "__main__":
    cli_main()
