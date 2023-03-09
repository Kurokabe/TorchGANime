# TorchGANime: Video generation of anime content conditioned on two frames

[Paper](./assets/GANime_paper.pdf) | [Presentation](https://docs.google.com/presentation/d/1KtN6-LmA6fbbY3wG6_Hz75HbOFL3J7vC/edit?usp=sharing&ouid=116500441313574364877&rtpof=true&sd=true)

**tl;dr** This is the PyTorch implementation of [GANime](https://github.com/Kurokabe/GANime), a model capable to generate video of anime content based on the first and last frame. This model is trained on a custom dataset based on the Kimetsu no Yaiba anime. It is composed of two model, a VQ-GAN for image generation, and a GPT2 transformer to generate the video frame by frame.

<p align="center">
    <img src=./assets/opu.jpg height="100" />
    <img src=./assets/omu.png height="100" />
    <img src=./assets/mse.png height="100" />
    <img src=./assets/hes-so.jpg height="100" />
</p>

This original project is a Master thesis realised by Farid Abdalla at HES-SO in partnership with Osaka Prefecture University (now renamed to Osaka Metropolitan University) in Japan  is available on [this repository](https://github.com/Kurokabe/GANime).

All implementation details are available in this [pdf](./assets/GANime_paper.pdf).

## Intermediate results
Here are some intermediate results obtained during the training of the model. The grey frame at the end is because the generated video depends on the longest video from the batch.

### Generated videos
![](./assets/results/tmp_generated.gif)

### Ground truth
![](./assets/results/tmp_real.gif)