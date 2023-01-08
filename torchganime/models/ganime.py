from typing import Union, Optional, List
import os
import torch
from transformers import get_scheduler
import pytorch_lightning as pl
from torchganime.models.video_transformer import (
    VideoTransformer,
    VideoTransformerConfig,
)
import deepspeed

# from torch.optim import AdamW
# from colossalai.nn.optimizer import HybridAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from loguru import logger


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec.to("cpu"), torch.zeros(*pad_size).to("cpu")], dim=dim)


class GANime(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        vqgan_ckpt_path: Union[str, os.PathLike],
        vocab_size: Optional[int] = None,
        n_positions: Optional[int] = None,
        n_embd: Optional[int] = None,
        n_layer: Optional[int] = None,
        n_head: Optional[int] = None,
        transformer_ckpt_path: Optional[Union[str, os.PathLike]] = None,
        use_position_embeddings: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        config = VideoTransformerConfig(
            vqgan_ckpt_path=vqgan_ckpt_path,
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            transformer_ckpt_path=transformer_ckpt_path,
            use_position_embeddings=use_position_embeddings,
        )
        self.video_transformer = VideoTransformer(config)
        self.video_transformer.gradient_checkpointing_enable()

    def forward(self, input, target=None):
        first_frames = input["first_frame"]
        end_frames = input["end_frame"]
        frame_number = input["frame_number"]
        return self.video_transformer(first_frames, end_frames, frame_number, target)

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self(input, target)
        loss = output["loss"]
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self.sample_videos_real = []
        self.sample_videos_gen = []

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self(input, target)
        loss = output["loss"]
        gen_video = output["video"]
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        if batch_idx < 4:
            if self.current_epoch == 0:
                self.sample_videos_real.append(target)  # .detach())
            self.sample_videos_gen.append(gen_video)  # .detach())

        return self.log_dict

    def preprocess_videos_for_logging(self, videos: List[torch.Tensor]):
        max_video_len = max([b.shape[2] for b in videos])
        videos = [pad_tensor(b, pad=max_video_len, dim=2) for b in videos]
        videos = torch.cat(videos, dim=0)
        videos = videos.permute(0, 2, 1, 3, 4)
        # videos values are between -1 and 1, normalize to 0 and 1
        videos = (videos + 1) / 2
        return videos

    def validation_epoch_end(self, outputs):
        # TODO remove pad_tensor to utils
        # TODO refactor this into a function
        if self.current_epoch == 0:
            real_videos = self.preprocess_videos_for_logging(self.sample_videos_real)
            self.logger.experiment.add_video(
                "videos_real",
                real_videos,
                self.current_epoch,
                fps=10,
            )

        rec_videos = self.preprocess_videos_for_logging(self.sample_videos_gen)
        self.logger.experiment.add_video(
            "videos_rec",
            rec_videos,
            self.current_epoch,
            fps=10,
        )

    def configure_optimizers(self):

        optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps(),
        )
        return [optimizer], [lr_scheduler]

    def num_training_steps(self) -> int:
        # TODO change this with a corrected version of the commented function below
        # num_steps = 14118
        num_steps = 1000
        num_epochs = 1000
        return num_steps * num_epochs

    # @property
    # def num_training_steps(self) -> int:
    #     """Total training steps inferred from datamodule and devices."""
    #     self.trainer.reset_train_dataloader()

    #     dataset = self.train_dataloader().loaders
    #     if self.trainer.max_steps:
    #         return self.trainer.max_steps

    #     dataset_size = (
    #         self.trainer.limit_train_batches
    #         if self.trainer.limit_train_batches != 0
    #         else len(dataset)
    #     )

    #     num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
    #     if self.trainer.tpu_cores:
    #         num_devices = max(num_devices, self.trainer.tpu_cores)

    #     effective_batch_size = (
    #         dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
    #     )
    #     return (dataset_size // effective_batch_size) * self.trainer.max_epochs
