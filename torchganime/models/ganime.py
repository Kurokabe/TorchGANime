from typing import List, Optional
from transformers import GPT2Config, PreTrainedModel, PretrainedConfig
from torchganime.models.vqgan import AutoencoderConfig
import pytorch_lightning as pl
from dataclasses import dataclass
from torchganime.models.vqgan import VQGAN
from transformers import GPT2LMHeadModel
import torch

import torch.nn.functional as F


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


class GANimeConfig(PretrainedConfig):
    model_type = "ganime"

    def __init__(
        self,
        vqgan_ckpt_path: str,
        transformer_config: GPT2Config,
        transformer_ckpt_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vqgan_ckpt_path = vqgan_ckpt_path
        self.transformer_config = transformer_config
        self.transformer_ckpt_path = transformer_ckpt_path


class GANime(PreTrainedModel):
    config_class = GANimeConfig

    def __init__(
        self,
        config: GANimeConfig,
    ):
        super().__init__(config)
        self.vqgan: VQGAN = VQGAN.load_from_checkpoint(config.vqgan_ckpt_path)
        self.transformer = GPT2LMHeadModel(config.transformer_config).from_pretrained(
            config.transformer_ckpt_path
        )

    def forward(self, tensor, labels=None):

        first_frames = tensor["first_frame"]
        end_frames = tensor["end_frame"]
        frame_number = tensor["frame_number"]

        if "target" in tensor:
            target = tensor["target"]

        # logits = self.model(tensor)
        # if labels is not None:
        #     loss = torch.nn.cross_entropy(logits, labels)
        #     return {"loss": loss, "logits": logits}
        # return {"logits": logits}

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
        for i in range(1, 5):  # frame_number.max()):
            remaining_frames = frame_number - i

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
                # print(logits.shape, target.shape)
                loss += F.cross_entropy(logits, current_target)

            return {"loss": loss, "video": generated_video}
        return {"video": generated_video}
