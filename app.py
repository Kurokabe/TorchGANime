import sys

sys.path.append("/TorchGANime/checkpoints/ganime_large/epoch=1222-step=14676.ckpt")

import torch
import gradio as gr
import numpy as np
import ffmpegio
import tempfile

from torchganime.models.ganime import GANime
from zero_to_fp32 import load_state_dict_from_zero_checkpoint


model = GANime(
    learning_rate=1e-5,
    vqgan_ckpt_path="/TorchGANime/checkpoints/vqgan_full/checkpoints/checkpoint.ckpt",
    transformer_ckpt_path="gpt2-large",
    use_token_type_ids=False,
    rec_loss_weight=0.0,
    perceptual_loss_weight=0.0,
)

# Initialize the DeepSpeed-Inference engine
model = load_state_dict_from_zero_checkpoint(
    model, "/TorchGANime/checkpoints/ganime_large/epoch=1222-step=14676.ckpt"
)
model = model.eval().cuda()


def normalize(image):
    image = image / 127.5 - 1

    return image


def save_video(video):
    output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

    filename = output_file.name
    video = video.numpy()
    # video = video * 255
    video = video.astype(np.uint8)
    ffmpegio.video.write(filename, 20, video, overwrite=True)
    return filename


def generate(first_image: np.ndarray, last_image: np.ndarray, num_frames: int):
    first_image = normalize(
        torch.from_numpy(first_image).permute(2, 0, 1).unsqueeze(0)
    ).cuda()
    last_image = normalize(
        torch.from_numpy(last_image).permute(2, 0, 1).unsqueeze(0)
    ).cuda()

    video = model.sample(first_image, last_image, num_frames)

    video = (video + 1) * 127.5

    video = video.permute(0, 2, 3, 4, 1).cpu()

    return save_video(video)


gr.Interface(
    generate,
    inputs=[
        gr.Image(label="Upload the first image"),
        gr.Image(label="Upload the last image"),
        gr.Slider(
            label="Number of frame to generate",
            minimum=15,
            maximum=100,
            value=15,
            step=1,
        ),
    ],
    outputs="video",
    title="Generate a video from the first and last frame",
).launch(share=True)
