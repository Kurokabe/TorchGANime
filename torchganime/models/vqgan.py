from dataclasses import dataclass
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .modules.diffusion.model import Decoder, Encoder
from .modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from .modules.vqvae.quantize import VectorQuantizer


@dataclass
class AutoencoderConfig:
    embed_dim: int
    n_embed: int
    channels: int
    z_channels: int
    resolution: int  # TODO modify with data params so that it is linked with the data
    in_channels: int
    out_channels: int
    ch_mult: List[int]
    n_res_blocks: int
    attn_resolutions: List[int]
    dropout: float
    resamp_with_conv: bool


@dataclass
class LossConfig:
    disc_channels: int
    disc_n_layers: int
    disc_use_actnorm: bool
    disc_start: int
    disc_factor: float
    disc_weight: float
    codebook_weight: float
    pixelloss_weight: float
    perceptual_weight: float
    disc_loss: str


class VQGAN(pl.LightningModule):
    def __init__(
        self,
        *,
        learning_rate: float,
        # embed_dim: int,
        # n_embed: int,
        # channels: int,
        # z_channels: int,
        # resolution: int,
        # in_channels: int = 3,
        # out_channels: int = 3,
        # ch_mult: List[int],
        # n_res_blocks: int,
        # attn_resolutions: List[int],
        # dropout: float,
        # resamp_with_conv: bool = True,
        autoencoder_config: AutoencoderConfig,
        loss_config: LossConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.encoder = Encoder(
            channels=autoencoder_config.channels,
            z_channels=autoencoder_config.z_channels,
            resolution=autoencoder_config.resolution,
            in_channels=autoencoder_config.in_channels,
            ch_mult=autoencoder_config.ch_mult,
            n_res_blocks=autoencoder_config.n_res_blocks,
            attn_resolutions=autoencoder_config.attn_resolutions,
            dropout=autoencoder_config.dropout,
            resamp_with_conv=autoencoder_config.resamp_with_conv,
        )
        self.decoder = Decoder(
            channels=autoencoder_config.channels,
            z_channels=autoencoder_config.z_channels,
            resolution=autoencoder_config.resolution,
            out_channels=autoencoder_config.out_channels,
            ch_mult=autoencoder_config.ch_mult,
            n_res_blocks=autoencoder_config.n_res_blocks,
            attn_resolutions=autoencoder_config.attn_resolutions,
            dropout=autoencoder_config.dropout,
            resamp_with_conv=autoencoder_config.resamp_with_conv,
        )

        self.quantize = VectorQuantizer(
            n_e=autoencoder_config.n_embed,
            e_dim=autoencoder_config.embed_dim,
            beta=0.25,
        )
        self.quant_conv = torch.nn.Conv2d(
            in_channels=autoencoder_config.z_channels,
            out_channels=autoencoder_config.embed_dim,
            kernel_size=1,
        )
        self.post_quant_conv = torch.nn.Conv2d(
            in_channels=autoencoder_config.embed_dim,
            out_channels=autoencoder_config.z_channels,
            kernel_size=1,
        )

        self.loss = VQLPIPSWithDiscriminator(
            disc_start=loss_config.disc_start,
            codebook_weight=loss_config.codebook_weight,
            pixelloss_weight=loss_config.pixelloss_weight,
            disc_num_layers=loss_config.disc_n_layers,
            disc_in_channels=autoencoder_config.out_channels,
            disc_factor=loss_config.disc_factor,
            disc_weight=loss_config.disc_weight,
            perceptual_weight=loss_config.perceptual_weight,
            use_actnorm=loss_config.disc_use_actnorm,
            disc_ndf=loss_config.disc_channels,
            disc_loss=loss_config.disc_loss,
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch):
        x = batch
        x = x.to(memory_format=torch.contiguous_format)
        return x.float()

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(
                qloss,
                x,
                xrec,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )

            self.log(
                "train/aeloss",
                aeloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(
                qloss,
                x,
                xrec,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            self.log(
                "train/discloss",
                discloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
            return discloss

    def on_validation_epoch_start(self) -> None:
        self.sample_images_real = []
        self.sample_images_rec = []

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(
            qloss,
            x,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        discloss, log_dict_disc = self.loss(
            qloss,
            x,
            xrec,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log(
            "val/rec_loss",
            rec_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/aeloss",
            aeloss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        if batch_idx < 4:
            self.sample_images_real.append(x.detach())
            self.sample_images_rec.append(xrec.detach())
        # grid_real = torchvision.utils.make_grid(x, normalize=True, value_range=(-1, 1))
        # grid_rec = torchvision.utils.make_grid(
        #     xrec, normalize=True, value_range=(-1, 1)
        # )

        # self.logger.experiment.add_image("images_real", grid_real, self.global_step)
        # self.logger.experiment.add_image("images_rec", grid_rec, self.global_step)

        # self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc, sync_dist=True)
        return self.log_dict

    def validation_epoch_end(self, outs) -> None:
        self.sample_images_real = torch.cat(self.sample_images_real, dim=0)
        grid_real = torchvision.utils.make_grid(
            self.sample_images_real, normalize=True, value_range=(-1, 1)
        )
        self.logger.experiment.add_image("images_real", grid_real, self.global_step)

        self.sample_images_rec = torch.cat(self.sample_images_rec, dim=0)
        grid_rec = torchvision.utils.make_grid(
            self.sample_images_rec, normalize=True, value_range=(-1, 1)
        )
        self.logger.experiment.add_image("images_rec", grid_rec, self.global_step)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantize.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
            eps=1e-6,
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr,
            betas=(0.5, 0.9),
            eps=1e-6,
        )

        # scheduler = ReduceLROnPlateau(
        #     opt_ae,
        #     "min",
        #     factor=0.5,
        #     patience=2,
        # )

        return [opt_ae, opt_disc], [
            # {
            #     "optimizer": opt_ae,
            #     "scheduler": scheduler,
            #     "monitor": "val/rec_loss_epoch",
            # }
        ]
