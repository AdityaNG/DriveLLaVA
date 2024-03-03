# DriveLLaVA

<img src="media/demo.gif">

<!-- [![codecov](https://codecov.io/gh/AdityaNG/DriveLLaVA/branch/main/graph/badge.svg?token=DriveLLaVA_token_here)](https://codecov.io/gh/AdityaNG/DriveLLaVA) -->
[![CI](https://github.com/AdityaNG/DriveLLaVA/actions/workflows/main.yml/badge.svg)](https://github.com/AdityaNG/DriveLLaVA/actions/workflows/main.yml)

Training LLaVA on the CommaVQ dataset to produce a tokenized trajectory

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## Dataset

Get started by downloading the CommaVQ Trajectory dataset from HuggingFace.
Then setup the DriveLLaVA repository.

```
cd ~/Datasets/
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/AdityaNG/commavq-trajectory ~/Datasets/commavq

cd ~/Datasets/commavq
git lfs pull
unzip "*.zip"

cd ~/
git clone https://github.com/AdityaNG/DriveLLaVA

cd ~/DriveLLaVA

sudo docker compose build

python3 -m drivellava.scripts.train
python3 -m drivellava.scripts.eval
```

## Generating trajectory templates

```bash
# Set the NUM_TRAJECTORY_TEMPLATES in drivellava/trajectory_encoder.py
python3 -m drivellava.scripts.generate_trajectory_templates
python3 -m drivellava.scripts.generate_sparse_llava_dataset_parallel
python3 -m drivellava.scripts.compile_jsons
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

Merge the model
```bash
cd LLaVA/
python scripts/merge_lora_weights.py \
  --model-path /path/to/lora_model \
  --model-base /path/to/base_model \
  --save-model-path
```

Setup the docker container for training
```bash
docker compose run dev
```

```bash
python3 -m drivellava.scripts.train
```


## TODO

- Training script
  - Select what layers to train
  - [Done] Measure memory requirements: ~ 40 GB vRAM
  - Train on CommaVQ
  - Tabulate results
- Dataset
  - [Done] Generate images from CommaVQ
  - [Done] Denoise the trajectory
  - [Done] Quantize the trajectory
  - [Done] Visualize the trajectory on the image
  - [Done] Generate JSON dataset

## References

- Fine-tuning LLaVA: https://ubiai.tools/how-to-fine-tune-llava-on-your-custom-dataset/