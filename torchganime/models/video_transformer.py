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


class VideoTransformer(PreTrainedModel):
    config_class = VideoTransformerConfig

    def __init__(
        self,
        config: VideoTransformerConfig,
    ):
        super().__init__(config)
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

    # def from_pretrained(
    #     self, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]]
    # ):
    #     self.transformer = self.transformer.from_pretrained(
    #         pretrained_model_name_or_path
    #     )

    def forward(self, first_frames, end_frames, frame_number, target=None):

        return self.predict(first_frames, end_frames, frame_number, target)

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

    def predict_next_indices(self, previous_frames_indices, end_frames_indices):
        previous_frames_indices = previous_frames_indices.reshape(
            end_frames_indices.shape
        )
        input = torch.cat((previous_frames_indices, end_frames_indices), dim=1)
        logits = self.transformer(input).logits
        # cut off conditioning
        logits = logits[:, end_frames_indices.shape[1] :]
        probs = F.softmax(logits, dim=-1)
        _, ix = torch.topk(probs, k=1, dim=-1)
        ix = torch.squeeze(ix)
        return ix, logits

    def predict(
        self, first_frames, end_frames, frame_number: torch.tensor, target=None
    ) -> dict:
        frame_number.max()
        first_frames_quant, first_frames_indices = self.encode(first_frames)
        _, end_frames_indices = self.encode(end_frames)

        generated_frames_indices = [first_frames_indices]
        predicted_logits = []
        for i in range(1, frame_number.max()):
            remaining_frames = (
                frame_number - i
            )  # TODO add remaining frames to input as positional encoding

            next_frame_indices, logits = self.predict_next_indices(
                generated_frames_indices[-1], end_frames_indices
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
            for i in range(len(predicted_logits)):
                _, _, target_info = self.vqgan.encode(target[:, :, i])
                target_indices = target_info[2]  # .view(target_quant.shape[0], -1)
                logits = predicted_logits[
                    i
                ]  # .reshape(-1, predicted_logits[i].size(-1))
                logits = logits.permute(0, 2, 1)
                current_target = target_indices.reshape(logits.shape[0], -1)
                loss += F.cross_entropy(logits, current_target)

            loss /= len(predicted_logits)
            return {"loss": loss, "video": generated_video}
        return {"video": generated_video}

    def gradient_checkpointing_enable(self):
        self.transformer.gradient_checkpointing_enable()
