FROM nvidia/cuda:11.8.0-devel-ubuntu20.04


ENV DEBIAN_FRONTEND noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y software-properties-common gcc
RUN apt-get install -y python3.10 python3.10-dev python3.10-distutils python3-pip python3-apt python3.10-venv
RUN apt-get update && apt-get install -y git curl
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN /usr/bin/python3.10 -m pip install --upgrade pip
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 # install pip

# Alias python3.10 to python3
RUN cp /usr/bin/python3.10 /usr/bin/python3

# Copy code in for installation
COPY ./ /app
WORKDIR /app

# make virtualenv 
SHELL ["/bin/bash", "-c"]

# Install LLaVA
WORKDIR /app/LLaVA

RUN pip install .

WORKDIR /app/

# RUN \
#     --mount=type=cache,target=/root/.cache/ \
#     make install

RUN make install

RUN python3 -m pip install flash-attn --no-build-isolation --no-cache-dir
RUN python3 -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall

# Install dependencies
# RUN /usr/bin/python3.10 -m pip install --upgrade pip
# RUN /usr/bin/python3.10 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117
# RUN /usr/bin/python3.10 -m pip install tqdm wandb opencv-python-headless pandas matplotlib==3.6.2 timm==0.6.12 scipy==1.9.3

# # Install requirements
# RUN /usr/bin/python3.10 -m pip install -r requirements.txt
# RUN /usr/bin/python3.10 -m pip install -r requirements-test.txt

# Remove code, without removing env
RUN find ./ ! -path './.venv*' ! -path './' -delete

# RUN mkdir /app
WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

ENV BNB_CUDA_VERSION 118

# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib

RUN echo 'PS1="(DLV) \[\]\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ \[\]"' >> /root/.bashrc
