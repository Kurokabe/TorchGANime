FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.12-cuda11.6.1
# Because of https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/ and https://github.com/NVIDIA/nvidia-docker/issues/1631#issuecomment-1112828208
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

# Update and install ffmpeg
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

# Setup environment
WORKDIR /TorchGANime
ENV PROJECT_DIR=/TorchGANime
COPY requirements.txt /TorchGANime/requirements.txt
RUN pip install -r requirements.txt
COPY . .
RUN pip install -e .
EXPOSE 8888