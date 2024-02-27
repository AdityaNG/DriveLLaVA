# DriveLLaVA

[![codecov](https://codecov.io/gh/AdityaNG/DriveLLaVA/branch/main/graph/badge.svg?token=DriveLLaVA_token_here)](https://codecov.io/gh/AdityaNG/DriveLLaVA)
[![CI](https://github.com/AdityaNG/DriveLLaVA/actions/workflows/main.yml/badge.svg)](https://github.com/AdityaNG/DriveLLaVA/actions/workflows/main.yml)

Training LLaVA on the CommaVQ dataset to produce a tokenized trajectory

## Install it from PyPI

```bash
pip install drivellava
```

## Usage

```py
from drivellava import BaseClass
from drivellava import base_function

BaseClass().base_method()
base_function()
```

```bash
$ python -m drivellava
#or
$ drivellava
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## Dataset

```
cd ~/Datasets/
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/AdityaNG/commavq-trajectory ~/Datasets/commavq

cd ~/Datasets/commavq
git lfs pull
unzip "*.zip"

cd ~/
git clone https://github.com/AdityaNG/DriveLLaVA

cd ~/DriveLLaVA

python3 -m drivellava.scripts.train
python3 -m drivellava.scripts.eval
```

## Running the scripts

```bash
python3 -m drivellava.scripts.generate_commavq_images
```

```bash
python3 -m drivellava.scripts.visualize_pose
```

```bash
python3 -m drivellava.scripts.generate_trajectory_templates
```

```bash
python3 -m drivellava.scripts.generate_sparse_llava_dataset
```

```bash
./scripts/extract_zips.sh ~/Datasets/commavq ~/Datasets/commavq

./scripts/compress_zips.sh ~/Datasets/commavq ~/Datasets/commavq-compressed
```

```bash
BNB_CUDA_VERSION=118 python3 -m drivellava.scripts.train
```

```bash
cd LLaVA

conda create -n llava python=3.10 -y
conda activate llava

pip install --upgrade pip  # enable PEP 660 support


conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install cudatoolkit=11.8 -c pytorch -c conda-forge

BNB_CUDA_VERSION=118
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aditya/miniconda3/envs/llava/lib
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib

pip install flash-attn --no-build-isolation --no-cache-dir

pip install .

```

## TODO

- Training script
  - Select what layers to train
  - Measure memory requirements
- Dataset
  - Generate images from CommaVQ
  - Denoise the trajectory
  - Quantize the trajectory
  - Visualize the trajectory on the image
  - Generate JSON dataset

## References

- Fine-tuning LLaVA: https://ubiai.tools/how-to-fine-tune-llava-on-your-custom-dataset/