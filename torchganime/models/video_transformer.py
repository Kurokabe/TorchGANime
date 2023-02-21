import os

from typing import List, Optional, Union
from transformers import GPT2Config, PreTrainedModel, PretrainedConfig
from torchganime.models.vqgan import AutoencoderConfig
import pytorch_lightning as pl
from dataclasses import dataclass
from torchganime.models.vqgan import VQGAN
from transformers import GPT2LMHeadModel
import torch
import math
from torch import nn, Tensor
import torch.nn.functional as F
from loguru import logger
from functools import cache
from torchganime.models.modules.losses.lpips import LPIPS

# class VQGANConfig(PretrainedConfig):
#     model_type = "vqgan"

#     def __init__(
#         self,
#         *,
#         embed_dim: int = 256,
#         n_embed: int = 50257,
#         channels: int = 256,
#         z_channels: int = 256,
#         resolution: int = 256,
#         in_channels: int = 3,
#         out_channels: int = 3,
#         ch_mult: List[int] = [1, 1, 2, 2],
#         n_res_blocks: int = 2,
#         attn_resolutions: List[int] = [32],
#         dropout: float = 0.0,
#         resamp_with_conv: bool = True,
#         ckpt_path: Optional[str] = None,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.n_embed = n_embed
#         self.channels = channels
#         self.z_channels = z_channels
#         self.resolution = resolution
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.ch_mult = ch_mult
#         self.n_res_blocks = n_res_blocks
#         self.attn_resolutions = attn_resolutions
#         self.dropout = dropout
#         self.resamp_with_conv = resamp_with_conv


class VideoTransformerConfig(PretrainedConfig):
    model_type = "videotransformer"

    def __init__(
        self,
        vqgan_ckpt_path: Union[str, os.PathLike] = "Kurokabe/VQGAN-KNY-SNK",
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        # transformer_config: GPT2Config = GPT2Config(),
        transformer_ckpt_path: Optional[str] = None,
        use_token_type_ids: bool = True,
        rec_loss_weight: float = 1.0,
        perceptual_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vqgan_ckpt_path = vqgan_ckpt_path
        # self.transformer_config = transformer_config
        # self.transformer_ckpt_path = transformer_ckpt_path
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.transformer_ckpt_path = transformer_ckpt_path
        self.use_token_type_ids = use_token_type_ids
        self.rec_loss_weight = rec_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight


class VideoTransformer(PreTrainedModel):
    config_class = VideoTransformerConfig

    def __init__(
        self,
        config: VideoTransformerConfig,
    ):
        super().__init__(config)
        self.config: VideoTransformerConfig
        self.vqgan: VQGAN = VQGAN.load_from_checkpoint(config.vqgan_ckpt_path)
        self.vqgan.eval()
        self.vqgan.requires_grad_(False)
        transformer_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.n_positions,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            use_cache=False,
        )
        if config.transformer_ckpt_path is None:
            self.transformer = GPT2LMHeadModel(transformer_config)
        else:
            self.transformer = GPT2LMHeadModel.from_pretrained(
                config.transformer_ckpt_path, use_cache=False
            )
            logger.info(
                f"""Loaded transformer from {config.transformer_ckpt_path} with 
            {self.transformer.config.vocab_size} tokens
            {self.transformer.config.n_positions} positions
            {self.transformer.config.n_embd} embedding dimensions
            {self.transformer.config.n_layer} layers
            {self.transformer.config.n_head} heads
            {self.transformer.num_parameters() / 1e6:.0f}M parameters"""
            )
        self.perceptual_loss = LPIPS().eval().requires_grad_(False)

    # def predict_next_indices(
    #     self,
    #     current_frames_indices: torch.tensor,
    #     end_frames_indices: torch.tensor,
    #     remaining_frames: torch.tensor,
    # ):
    #     remaining_frames = remaining_frames.view(1, -1).T.repeat(
    #         1, end_frames_indices.shape[1]
    #     )

    #     input = torch.stack(
    #         (
    #             remaining_frames,
    #             end_frames_indices,
    #             current_frames_indices,
    #         ),
    #         dim=1,
    #     )
    #     if self.config.use_token_type_ids:
    #         token_type_ids = torch.zeros_like(input)
    #         token_type_ids[:, 1:] = 1
    #     else:
    #         token_type_ids = None
    #     logits = self.transformer(input, token_type_ids=token_type_ids).logits

    #     # cut off conditioning
    #     logits = logits[:, -1]
    #     probs = F.softmax(logits, dim=-1)
    #     _, ix = torch.topk(probs, k=1, dim=-1)
    #     next_frame_indices = torch.squeeze(ix)
    #     return next_frame_indices, logits

    def get_visual_loss(
        self, target_frame: torch.tensor, predicted_frame: torch.tensor
    ):
        if self.config.rec_loss_weight > 0:
            rec_loss = torch.abs(
                target_frame.contiguous() - predicted_frame.contiguous()
            ).mean()
        else:
            rec_loss = 0.0

        if self.config.perceptual_loss_weight > 0:
            p_loss = self.perceptual_loss(
                target_frame.contiguous(), predicted_frame.contiguous()
            ).mean()
        else:
            p_loss = 0.0

        nll_loss = rec_loss + p_loss
        return nll_loss, rec_loss, p_loss

    def forward(
        self,
        first_frame: torch.tensor,
        end_frame: torch.tensor,
        frame_number: torch.tensor,
        target: torch.tensor = None,
    ):
        return self.predict(first_frame, end_frame, frame_number, target)

    # def forward(
    #     self,
    #     current_frames: torch.tensor,
    #     end_frames: torch.tensor,
    #     remaining_frames: torch.tensor,
    #     target: torch.tensor = None,
    # ):

    # current_frames_quant, current_frames_indices = self.encode(current_frames)
    # _, end_frames_indices = self.encode(end_frames)

    # next_frame_indices, logits = self.predict_next_indices(
    #     current_frames_indices, end_frames_indices, remaining_frames
    # )

    # next_frame = self.decode_to_img(next_frame_indices, current_frames_quant.shape)

    # if target is not None:
    #     _, target_indices = self.encode(target)
    #     logits = logits.permute(0, 2, 1)
    #     target_indices = target_indices.reshape(logits.shape[0], -1)
    #     scce_loss = F.cross_entropy(logits, target_indices)
    #     nll_loss, rec_loss, p_loss = self.get_visual_loss(target, next_frame)

    #     loss = scce_loss + nll_loss
    #     return {
    #         "next_frame": next_frame,
    #         "loss": loss,
    #         "scce_loss": scce_loss,
    #         "nll_loss": nll_loss,
    #         "rec_loss": rec_loss,
    #         "p_loss": p_loss,
    #     }

    # return {"next_frame": next_frame}

    @torch.no_grad()
    def encode(self, frame):
        quant_z, _, info = self.vqgan.encode(frame)
        indices = info[2].view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
        quant_z = self.vqgan.quantize.get_codebook_entry(index.reshape(-1), shape=bhwc)
        x = self.vqgan.decode(quant_z)
        return x

    def predict_next_indices(
        self,
        previous_frames_indices,
        end_frames_indices,
        current_frame_number,
        frame_number,
    ):
        previous_frames_indices = previous_frames_indices.reshape(
            end_frames_indices.shape
        )

        remaining_frames = (
            (frame_number - current_frame_number)
            .view(1, -1)
            .T.repeat(1, end_frames_indices.shape[1])
        )

        input = torch.stack(
            (remaining_frames, previous_frames_indices, end_frames_indices), dim=1
        )
        # attention_mask = self.get_attention_mask(
        #     input, current_frame_number, frame_number
        # )

        logits = self.transformer(input).logits
        # cut off conditioning
        logits = logits[:, -1]
        probs = F.softmax(logits, dim=-1)
        _, ix = torch.topk(probs, k=1, dim=-1)
        ix = torch.squeeze(ix)
        return ix, logits

    # def compute_position_ids(
    #     self, current_frame_number, frame_number, frame_indices_shape
    # ):
    #     position_id_end_frame = frame_number.view(1, -1).T.repeat(
    #         1, frame_indices_shape
    #     )
    #     position_id_current_frame = (
    #         torch.ones_like(position_id_end_frame) * current_frame_number
    #     )
    #     position_ids = torch.stack(
    #         [position_id_current_frame, position_id_end_frame], dim=1
    #     )
    #     return position_ids

    # def get_attention_mask(self, transformer_input, current_frame_number, frame_number):
    #     # Get for each batch element a value, 0 or 1 for the padding. 1 means not padded frame, 0 means padded frame
    #     padded_mask = (frame_number > current_frame_number).type(torch.int)
    #     target_shape = transformer_input.shape
    #     # Expand the mask to the shape of the input
    #     return (
    #         padded_mask.view(len(padded_mask), *(1,) * (len(target_shape) - 1))
    #         .expand(target_shape)
    #         .contiguous()
    #     )

    def predict(
        self, first_frames, end_frames, frame_number: torch.tensor, target=None
    ) -> dict:

        first_frames_quant, first_frames_indices = self.encode(first_frames)
        _, end_frames_indices = self.encode(end_frames)

        generated_frames_indices = [first_frames_indices]
        predicted_logits = []
        for i in range(1, frame_number.max()):

            if target is not None and self.training:
                _, previous_frame_indices = self.encode(target[:, :, i - 1])
            else:
                previous_frame_indices = generated_frames_indices[-1]

            next_frame_indices, logits = self.predict_next_indices(
                previous_frame_indices,
                end_frames_indices,
                current_frame_number=i,
                frame_number=frame_number,
            )
            predicted_logits.append(logits)
            generated_frames_indices.append(next_frame_indices)

        generated_video = [
            self.decode_to_img(frame_indices, first_frames_quant.shape)
            for frame_indices in generated_frames_indices
        ]
        generated_video = torch.stack(generated_video, dim=2)

        if target is not None:
            loss = 0
            scce_loss = 0
            rec_loss = 0
            p_loss = 0
            for i in range(len(predicted_logits)):
                _, _, target_info = self.vqgan.encode(target[:, :, i])
                target_indices = target_info[2]  # .view(target_quant.shape[0], -1)
                logits = predicted_logits[
                    i
                ]  # .reshape(-1, predicted_logits[i].size(-1))
                logits = logits.permute(0, 2, 1)
                current_target = target_indices.reshape(logits.shape[0], -1)
                scce_loss += F.cross_entropy(logits, current_target)

            scce_loss /= len(predicted_logits)

            for i in range(len(generated_video)):
                rec_loss += torch.abs(
                    target[:, :, i].contiguous() - generated_video[:, :, i].contiguous()
                )
                p_loss += self.perceptual_loss(
                    target[:, :, i].contiguous(), generated_video[:, :, i].contiguous()
                )

            nll_loss = rec_loss + p_loss
            nll_loss /= len(generated_video)
            # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            nll_loss = torch.mean(nll_loss)
            loss = scce_loss + nll_loss

            return {
                "loss": loss,
                "scce_loss": scce_loss,
                "nll_loss": nll_loss,
                "video": generated_video,
            }
        return {"video": generated_video}

    def gradient_checkpointing_enable(self):
        self.transformer.gradient_checkpointing_enable()
