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
import torchvision

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
        use_token_type_ids: bool = True,
        rec_loss_weight: float = 1.0,
        perceptual_loss_weight: float = 1.0,
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
            use_token_type_ids=use_token_type_ids,
            rec_loss_weight=rec_loss_weight,
            perceptual_loss_weight=perceptual_loss_weight,
        )
        self.video_transformer = VideoTransformer(config)
        self.video_transformer.gradient_checkpointing_enable()

    def forward(self, input, target=None):
        current_frames = input["current_frames"]
        end_frames = input["end_frames"]
        remaining_frames = input["remaining_frames"]
        return self.video_transformer(
            current_frames, end_frames, remaining_frames, target
        )

    @torch.no_grad()
    def sample(self, first_frames, end_frames, frame_number):
        batch_size = first_frames.shape[0]
        frame_number = torch.tensor(frame_number).repeat(batch_size).to(self.device)
        generated_video = [first_frames]
        for i in range(1, frame_number.max()):
            output = self.video_transformer(
                generated_video[-1], end_frames, frame_number - i
            )
            generated_video.append(output["next_frame"])
        return torch.stack(generated_video, dim=2)

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
        other_losses = {
            f"train/{key}": value for key, value in output.items() if "_loss" in key
        }
        self.log_dict(
            other_losses, prog_bar=False, logger=True, on_step=True, on_epoch=True
        )

        return loss

    def on_validation_epoch_start(self) -> None:
        self.sample_videos_real = []
        self.sample_videos_gen = []
        self.current_frames = []
        self.end_frames = []
        self.predicted_frames = []
        self.target_frames = []

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self(input, target)
        loss = output["loss"]
        # gen_video = output["video"]
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        other_losses = {
            f"val/{key}": value for key, value in output.items() if "_loss" in key
        }
        self.log_dict(
            other_losses, prog_bar=False, logger=True, on_step=True, on_epoch=True
        )
        if batch_idx < 1:
            self.sample_videos_gen.append(
                self.sample(
                    input["current_frames"], input["end_frames"], frame_number=15
                )
            )

            self.current_frames.append(input["current_frames"])
            self.end_frames.append(input["end_frames"])
            self.predicted_frames.append(output["next_frame"])
            self.target_frames.append(target)
        #     # if self.current_epoch == 0:
        #     self.sample_videos_real.append(target)  # .detach())
        #     self.sample_videos_gen.append(gen_video)  # .detach())

        return self.log_dict

    def preprocess_videos_for_logging(self, videos: List[torch.Tensor]):
        max_video_len = max([b.shape[2] for b in videos])
        videos = [pad_tensor(b, pad=max_video_len, dim=2) for b in videos]
        videos = torch.cat(videos, dim=0)
        videos = videos.permute(0, 2, 1, 3, 4)
        # videos values are between -1 and 1, normalize to 0 and 1
        videos = (videos + 1) / 2
        return videos

    def log_images(
        self,
        current_frames: List[torch.tensor],
        end_frames: List[torch.tensor],
        predicted_frames: List[torch.tensor],
        target_frames: List[torch.tensor],
        name: str,
    ):
        current_frames = torch.cat(current_frames, dim=0)
        end_frames = torch.cat(end_frames, dim=0)
        predicted_frames = torch.cat(predicted_frames, dim=0)
        target_frames = torch.cat(target_frames, dim=0)

        current_frames_grid = torchvision.utils.make_grid(
            current_frames, nrow=1, normalize=True, value_range=(-1, 1)
        )
        end_frames_grid = torchvision.utils.make_grid(
            end_frames, nrow=1, normalize=True, value_range=(-1, 1)
        )
        predicted_frames_grid = torchvision.utils.make_grid(
            predicted_frames, nrow=1, normalize=True, value_range=(-1, 1)
        )
        target_grid = torchvision.utils.make_grid(
            target_frames, nrow=1, normalize=True, value_range=(-1, 1)
        )

        grid = torch.cat(
            (current_frames_grid, end_frames_grid, predicted_frames_grid, target_grid),
            dim=2,
        )
        self.logger.experiment.add_image(name, grid, self.current_epoch)

    def validation_epoch_end(self, outputs):
        # TODO remove pad_tensor to utils
        # TODO refactor this into a function
        # if self.current_epoch == 0:
        #     real_videos = self.preprocess_videos_for_logging(self.sample_videos_real)
        #     self.logger.experiment.add_video(
        #         "videos_real",
        #         real_videos,
        #         self.current_epoch,
        #         fps=10,
        #     )

        rec_videos = self.preprocess_videos_for_logging(self.sample_videos_gen)
        self.logger.experiment.add_video(
            "videos_rec",
            rec_videos,
            self.current_epoch,
            fps=10,
        )

        self.log_images(
            self.current_frames,
            self.end_frames,
            self.predicted_frames,
            self.target_frames,
            "val/current_end_predicted_target",
        )

    def configure_optimizers(self):

        optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer,
            num_warmup_steps=int(self.trainer.estimated_stepping_batches * 0.15),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return [optimizer], [lr_scheduler]

    def num_training_steps(self) -> int:
        # TODO change this with a corrected version of the commented function below
        # num_steps = 14118
        num_steps = 100
        num_epochs = 200
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
